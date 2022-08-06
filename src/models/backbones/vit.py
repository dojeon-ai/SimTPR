import torch.nn as nn
import torch
import numpy as np
from .base import BaseBackbone
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from src.common.train_utils import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed, get_1d_masked_input, get_3d_masked_input


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
        if attn_mask:
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


class VIT(BaseBackbone):
    name = 'vit'
    def __init__(self,
                 obs_shape,
                 action_size,
                 patch_size,
                 t_step,
                 pool,
                 enc_depth,
                 enc_dim, 
                 enc_mlp_dim,
                 enc_heads, 
                 dec_depth, 
                 dec_dim, 
                 dec_mlp_dim,
                 dec_heads, 
                 emb_dropout,
                 dropout):

        super().__init__()
        frame, channel, image_height, image_width = obs_shape
        image_channel = frame * channel
        patch_height, patch_width = patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = image_channel * patch_height * patch_width

        self.t_step = t_step
        self.num_patches = num_patches

        assert pool in {'mean'}, 'currently, pool must be mean (mean pooling)'

        ###########################################
        # Encoder 
        self.patch_embed = nn.Sequential(
            Rearrange('n t c (h p1) (w p2) -> n (t h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, enc_dim),
        )

        self.enc_spatial_embed = nn.Parameter(torch.randn(1, num_patches, enc_dim), requires_grad=False)
        self.enc_temporal_embed = nn.Parameter(torch.randn(1, t_step, enc_dim), requires_grad=False)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.encoder = Transformer(dim=enc_dim, 
                                   depth=enc_depth, 
                                   heads=enc_heads, 
                                   mlp_dim=enc_mlp_dim, 
                                   dropout=dropout)
        self.enc_norm = nn.LayerNorm(enc_dim)

        ########################################
        # Decoder
        self.decoder_embed = nn.Linear(enc_dim, dec_dim)
        self.act_embed = nn.Embedding(action_size, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))

        self.dec_spatial_embed = nn.Parameter(torch.randn(1, num_patches, dec_dim), requires_grad=False) 
        self.dec_temporal_embed = nn.Parameter(torch.randn(1, t_step, dec_dim), requires_grad=False)
        self.dec_emb_dropout = nn.Dropout(emb_dropout)
        
        self.decoder = Transformer(dim=dec_dim, 
                                   depth=dec_depth, 
                                   heads=dec_heads, 
                                   mlp_dim=dec_mlp_dim, 
                                   dropout=dropout)
        self.dec_norm = nn.LayerNorm(dec_dim)
        
        self.patch_pred = nn.Linear(dec_dim, patch_dim, bias=True)
        self.act_pred = nn.Linear(dec_dim, action_size, bias=True)
        
        self._output_dim = dec_dim
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) spatial pos_embed by 2d sin-cos embedding
        # initialize (and freeze) temporal pos_embed by 1d sin-cos embedding
        enc_spatial_embed = get_2d_sincos_pos_embed(self.enc_spatial_embed.shape[-1], int((self.enc_spatial_embed.shape[1])**.5))
        self.enc_spatial_embed.copy_(torch.from_numpy(enc_spatial_embed).float().unsqueeze(0))

        enc_temporal_embed = get_1d_sincos_pos_embed_from_grid(self.enc_temporal_embed.shape[-1], np.arange(int(self.enc_temporal_embed.shape[1])))
        self.enc_temporal_embed.copy_(torch.from_numpy(enc_temporal_embed).float().unsqueeze(0))

        dec_spatial_embed = get_2d_sincos_pos_embed(self.dec_spatial_embed.shape[-1], int((self.dec_spatial_embed.shape[1])**.5))
        self.dec_spatial_embed.copy_(torch.from_numpy(dec_spatial_embed).float().unsqueeze(0))

        dec_temporal_embed = get_1d_sincos_pos_embed_from_grid(self.dec_temporal_embed.shape[-1], np.arange(int(self.dec_temporal_embed.shape[1])))
        self.dec_temporal_embed.copy_(torch.from_numpy(dec_temporal_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def get_decoder_attn_mask(self, done):
        N, T = done.shape
        
        # get uni-directional attn_mask (1: mask-out, 0: leave)
        L = self.t_step * (self.num_patches+1)
        attn_mask = 1 - torch.ones((N, L, L), device=done.device).tril_()
        
        # find indexs where done is True
        done = done.float()
        done_mask = torch.zeros_like(done)
        done_idx = torch.nonzero(done==1)
        
        # done-mask (1: mask_out, 0: leave).
        # done is masked in reverse-order is required to keep consistency with evaluation stage.
        for idx in done_idx:
            row = idx[0]
            col = idx[1]
            done_mask[row, :col+1] = 1
            
        # repeat for patches & actions
        done_mask = torch.repeat_interleave(done_mask, repeats=self.num_patches+1, dim=1)
            
        # expand to attn_mask
        done_mask = 1 -(1-done_mask).unsqueeze(-1).matmul((1-done_mask).unsqueeze(1))
        
        # 0: attn_mask & done_mask are both 0
        attn_mask = 1 - ((attn_mask == 0) * (done_mask == 0)).float()
        
        return attn_mask
        
        

    def forward(self, x, input_mask=None):
        """
        [param] x: dict
            img: (N, T, C, H, W)
            act: (N, T) 
            done: (N, T)
            img_mask: (N, T)
        [param] input_mask: dict
            patch_mask_type
            patch_ids_keep
            patch_ids_restore
            act_ids_keep
            act_ids_restore
        """
        img = x['img']
        act = x['act']
        done = x['done']
        
        N, T, C, H, W = img.shape

        # TODO: model eval시에는 어떻게하면 masking되지 않고 진행?
        # TODO: done 관련 처리 restore x
        # TODO: attn_mask
        
        ##############################################
        # Encoder
        
        # N, L=T*P, D (T: t_step P: num_patches)
        x = self.patch_embed(img)

        # add pos embed w/o act token
        # pos_embed = spatial_embed + temporal_embed
        enc_spatial_embed = self.enc_spatial_embed.repeat(1, self.t_step, 1)
        enc_temporal_embed = torch.repeat_interleave(self.enc_temporal_embed, repeats=self.num_patches, dim=1)
        enc_pos_embed = enc_spatial_embed + enc_temporal_embed
        x = x + enc_pos_embed

        # masking: length -> length * mask_ratio
        if input_mask:
            x = rearrange(x, 'n (t p) d -> n t p d', t = self.t_step, p = self.num_patches)
            x = get_3d_masked_input(x, input_mask['patch_ids_keep'], input_mask['patch_mask_type'])
            
        # apply Transformer blocks
        x = self.emb_dropout(x)
        x = self.encoder(x)
        x = self.enc_norm(x)

        ##############################################
        # Decoder
        
        # embed patches
        self.decoder_embed(x)
        
        # restore patch-mask
        if input_mask:
            patch_mask_len = self.t_step * self.num_patches - x.shape[1]
            mask_tokens = torch.zeros((N, patch_mask_len, x.shape[-1])).to(x.device)
            x = torch.cat([x, mask_tokens], dim=1)
            x = torch.gather(x, dim=1, index=input_mask['patch_ids_restore'].unsqueeze(-1).repeat(1,1,x.shape[-1]))
        
        # pos-embed to patches
        dec_spatial_embed = self.dec_spatial_embed.repeat(1, self.t_step, 1)
        dec_temporal_embed = torch.repeat_interleave(self.dec_temporal_embed, repeats=self.num_patches, dim=1)
        dec_pos_embed = dec_spatial_embed + dec_temporal_embed
        x = x + dec_pos_embed
        
        # embed actions
        x_act = self.act_embed(act)
        
        # mask & restore actions
        if input_mask:
            x_act = get_1d_masked_input(x_act, input_mask['act_ids_keep'])            
            act_mask_len = self.t_step - x_act.shape[1]
            act_mask_tokens = torch.zeros((N,act_mask_len,x_act.shape[-1])).to(x.device)
            x_act = torch.cat([x_act, act_mask_tokens], dim=1)
            x_act = torch.gather(x_act, dim=1, index=input_mask['act_ids_restore'].unsqueeze(-1).repeat(1,1,x_act.shape[-1]))
        
        
        # pos-embed to actions
        x_act = x_act + self.dec_temporal_embed
        
        # concat patches with actions
        x = rearrange(x, 'n (t p) d -> n t p d', t = self.t_step, p = self.num_patches)
        x = torch.cat([x, x_act.unsqueeze(2)], dim=2)
        x = rearrange(x, 'n t pa d -> n (t pa) d', t = self.t_step, pa = self.num_patches+1) # +1 for act

        # casual attention mask
        attn_mask = self.get_decoder_attn_mask(done)

        # decoder
        x = self.dec_emb_dropout(x)
        x = self.decoder(x, attn_mask)
        x = self.dec_norm(x)        
        
        # extract patch & act
        patch_ids = torch.arange(P*T, device=x.device).reshape(T, P) + torch.arange(T, device=x.device).reshape(-1,1)
        x_patch = torch.gather(x, dim=1, index=patch_ids.reshape(1,-1,1).repeat(N,1,D))
        
        act_ids = (torch.arange(T, device=x.device)+1) * (P+1) - 1
        x_act = torch.gather(x, dim=1, index=act_ids.reshape(1,-1,1).repeat(N,1,D))
        
        # prediction layer
        patch_pred = self.patch_pred(x_patch)
        act_pred = self.act_pred(x_act)

        return patch_pred, act_pred, patch_mask, act_mask
