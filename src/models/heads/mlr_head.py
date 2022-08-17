import torch.nn as nn
import torch
from .base import BaseHead
import numpy as np
from src.common.train_utils import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed, get_1d_masked_input, get_3d_masked_input
from einops import rearrange, repeat
from src.models.layers import Transformer

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

class MLRHead(BaseHead):
    name = 'mlr'
    def __init__(self,
                 patch_size,
                 action_size,
                 embedding_output_dim,
                 projection_dim,
                 latent_decoder_depth,
                 latent_decoder_heads,
                 dropout,
                 hid_features,
                 out_features,
                 emb_dropout,
                 t_step):
        super().__init__()

        self.embedding_output_dim = embedding_output_dim
        self.projection_dim = projection_dim
        self.t_step = t_step
        latent_decoder_mlp_dim = self.projection_dim * 4
        self.patch_size = patch_size
        self.action_size = action_size

        # Projection to Transformer input dimension
        self.projection = nn.Linear(self.embedding_output_dim, self.projection_dim)
        self.action = nn.Embedding(self.action_size, self.projection_dim)

        self.patch_mask_token = nn.Parameter(torch.zeros(1, 1, self.projection_dim))
        self.act_mask_token = nn.Parameter(torch.zeros(1, 1, self.projection_dim))
        
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.latent_decoder_positional = nn.Parameter(torch.randn(1, t_step, self.projection_dim), requires_grad=False)
        self.latent_decoder = Transformer(dim=self.projection_dim, 
                                   depth=latent_decoder_depth, 
                                   heads=latent_decoder_heads, 
                                   mlp_dim=latent_decoder_mlp_dim, 
                                   dropout=dropout)

        self.projector = nn.Sequential(  
            nn.Linear(in_features=self.projection_dim, out_features=hid_features),
            nn.BatchNorm1d(num_features=hid_features),
            nn.ReLU(),
            nn.Linear(in_features=hid_features, out_features=out_features)
        )

        self.predictor = nn.Sequential(
            nn.Linear(in_features=out_features, out_features=out_features)
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
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def project(self, x):
        
        x = self.projector(x)

        return x

    def predict(self, x):
        x = self.predictor(x)
        return x

    def forward(self, x, use_action=False):

        # get input
        patch = self.projection(x['patch'])
        act = self.action(x['act'])
        done = x['done']

        # add positional_embedding
        x = patch + self.latent_decoder_positional

        x_act = act + self.latent_decoder_positional[:, :-1, :]

        done = done.float()
        done_idx = torch.nonzero(done==1)
        for idx in done_idx:
            row = idx[0]
            col = idx[1]
            done[row, :col+1] =  1

        x_act = x_act * (1 - done.unsqueeze(-1))

        # import pdb; pdb.set_trace()

        x = torch.cat((x, x_act), dim=1)

        x = self.emb_dropout(x)

        x = self.latent_decoder(x)

        # use only states
        x = x[:, :self.t_step, :]

        x = rearrange(x, 'n t d -> (n t) d') 

        x = self.project(x)
        x = self.predict(x)
        
        return x