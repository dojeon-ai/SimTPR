import torch.nn as nn
import torch
from .base import BaseHead
import numpy as np
from src.common.train_utils import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed, get_1d_masked_input, get_3d_masked_input
from einops import rearrange, repeat


def get_attn_mask(t_step, num_patches, done, use_action):
    """
    [params] t_step: time step 
    [params] num_patches: number of patches for each time step
    [params] done: (N, T-1)
    [params] use_action: (bool) whether to attend action or not
    [returns] attn_mask: (N, L, L) (L=T*(P+1)-1)
    """
    N = done.shape[0]
    L = t_step * num_patches

    # find indexs where done is True
    done_mask = torch.zeros((N, t_step), device=done.device)
    done = done.float()
    done_idx = torch.nonzero(done==1)

    # done-mask (1: mask_out, 0: leave).
    # done is masked in reverse-order is required to keep consistency with evaluation stage.
    for idx in done_idx:
        row = idx[0]
        col = idx[1]
        done_mask[row, :col+1] = 1

    # repeat for patches
    patch_mask = torch.repeat_interleave(done_mask, repeats=num_patches, dim=1)

    # expand to attn_mask
    attn_mask = 1 -(1-patch_mask).unsqueeze(-1).matmul((1-patch_mask).unsqueeze(1))

    # generate additional mask if use_action
    if use_action:
        act_mask = done_mask[:, :-1]     
        
    else:
        act_mask = torch.ones((N, t_step-1), device=done.device)

    # generate patch-act mask
    patch_act_mask = 1 -(1-patch_mask).unsqueeze(-1).matmul((1-act_mask).unsqueeze(1))
    act_patch_mask = 1 -(1-act_mask).unsqueeze(-1).matmul((1-patch_mask).unsqueeze(1))
    act_act_mask = 1 -(1-act_mask).unsqueeze(-1).matmul((1-act_mask).unsqueeze(1))
        
    # concatenate patch-act mask to attn_mask
    attn_mask = torch.cat([attn_mask, patch_act_mask], dim=2)
    _act_mask = torch.cat([act_patch_mask, act_act_mask], dim=2)
    attn_mask = torch.cat([attn_mask, _act_mask], dim=1)
        
    return attn_mask

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 2, dropout = 0.):
        super().__init__()
        head_dim = dim // heads
        self.heads = heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) 

    def forward(self, x, attn_mask=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'n t (h d) -> n h t d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        # attn_mask: (n, t, t)
        if attn_mask is not None:
            attn = attn * (1-attn_mask).unsqueeze(1)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'n h t d -> n t (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, attn_mask=None):
        for attn, ff in self.layers:
            x = attn(x, attn_mask=attn_mask) + x
            x = ff(x) + x
        return x

class MLRHead(BaseHead):
    name = 'mlr'
    def __init__(self,
                 patch_size,
                 action_size,
                 latent_decoder_depth,
                 latent_decoder_heads,
                 dropout,
                 hid_features,
                 out_features,
                 t_step):
        super().__init__()

        self.embedding_output_dim = 3136
        self.projection_dim = 256
        latent_decoder_mlp_dim = 256 * 4
        self.patch_size = patch_size
        self.action_size = action_size

        self.projection = nn.Linear(self.embedding_output_dim, self.projection_dim)
        self.action = nn.Embedding(self.action_size, self.projection_dim)
        
        self.latent_decoder_positional = nn.Parameter(torch.randn(1, t_step, self.projection_dim), requires_grad=False)
        self.latent_decoder = Transformer(dim=self.projection_dim, 
                                   depth=latent_decoder_depth, 
                                   heads=latent_decoder_heads, 
                                   mlp_dim=latent_decoder_mlp_dim, 
                                   dropout=dropout)

        self.projector_Linear = nn.Linear(in_features=self.projection_dim, out_features=hid_features)
        self.projector_Batch_norm = nn.BatchNorm1d(num_features=hid_features)
        self.projector_ReLU = nn.ReLU()
        self.projector_Linear2 = nn.Linear(in_features=hid_features, out_features=out_features)

        self.predictor = nn.Sequential(
            nn.Linear(in_features=out_features, out_features=out_features),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # initialize (and freeze) pos_embed by 1d sin-cos embedding
        latent_decoder_positional = get_1d_sincos_pos_embed_from_grid(self.latent_decoder_positional.shape[-1], np.arange(int(self.latent_decoder_positional.shape[1])))
        self.latent_decoder_positional.copy_(torch.from_numpy(latent_decoder_positional).float().unsqueeze(0))

        # initialize nn.Conv2d, nn.Linear, and nn.LayerNorm
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def project(self, x):
        x = self.projector_Linear(x)
        x = x.transpose(1, 2)
        x = self.projector_Batch_norm(x)
        x = x.transpose(1, 2)
        x = self.projector_ReLU(x)
        x = self.projector_Linear2(x)
        return x

    def predict(self, x):
        x = self.predictor(x)
        return x

    def forward(self, x):
        #import pdb; pdb.set_trace()
        patch = x['patch']
        act = x['act']

        x = patch + self.latent_decoder_positional

        x_act = act + self.latent_decoder_positional

        x_in = torch.zeros(x.shape[0], x.shape[1] *2, x.shape[2]).to(patch.device)

        x_in[:, ::2, :] = x
        x_in[:, 1::2, :] = x_act

        x = self.latent_decoder(x_in)

        x = self.project(x)
        x = self.predict(x)
        
        return x