import torch.nn as nn
import torch
import numpy as np
from einops import rearrange, repeat
from .base import BaseHead
from src.models.layers import Transformer
from src.common.train_utils import xavier_uniform_init
from src.common.vit_utils import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed
from src.common.vit_utils import get_1d_masked_input, get_3d_masked_input


class MAEHead(BaseHead):
    name = 'mae_head'
    def __init__(self, 
                 obs_shape,
                 action_size,
                 patch_size,
                 process_type,
                 t_step,
                 in_dim,
                 dec_depth, 
                 dec_dim, 
                 dec_mlp_dim,
                 dec_heads, 
                 emb_dropout,
                 dropout):
        super().__init__()
        t, c, i_h, i_w = obs_shape
        
        assert process_type in {'indiv_frame', 'stack_frame'}
        self.process_type = process_type
        if process_type == 'indiv_frame':
            image_channel = c
        elif process_type == 'stack_frame':
            image_channel = t * c
   
        p_h, p_w = patch_size
        assert i_h % p_h == 0 and i_w % p_w == 0, 'Image must be divisible by the patch size.'
        num_patches = (i_h // p_h) * (i_w // p_w)
        patch_dim = image_channel * p_h * p_w
        
        self.t_step = t_step
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.dec_dim = dec_dim
        
        self.decoder_embed = nn.Linear(in_dim, dec_dim)                
        self.patch_mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.spatial_embed = nn.Parameter(torch.randn(1, num_patches, dec_dim), requires_grad=False) 
        self.temporal_embed = nn.Parameter(torch.randn(1, t_step, dec_dim), requires_grad=False)
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        self.decoder = Transformer(dim=dec_dim, 
                                   depth=dec_depth, 
                                   heads=dec_heads, 
                                   mlp_dim=dec_mlp_dim, 
                                   dropout=dropout)
        self.out_norm = nn.LayerNorm(dec_dim)        
        self.patch_pred = nn.Linear(dec_dim, patch_dim, bias=True)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        # initialize (and freeze) spatial pos_embed by 2d sin-cos embedding
        # initialize (and freeze) temporal pos_embed by 1d sin-cos embedding=
        T, P, D = self.t_step, self.num_patches, self.dec_dim

        dec_spatial_embed = get_2d_sincos_pos_embed(D, int((P)**.5))
        self.spatial_embed.copy_(torch.from_numpy(dec_spatial_embed).float().unsqueeze(0))
        self.spatial_embed.requires_grad = False

        dec_temporal_embed = get_1d_sincos_pos_embed_from_grid(D, np.arange(int(T)))
        self.temporal_embed.copy_(torch.from_numpy(dec_temporal_embed).float().unsqueeze(0))
        self.temporal_embed.requires_grad = False

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.patch_mask_token, std=.02)
        
        # initialize nn.Linear and nn.LayerNorm
        self.apply(xavier_uniform_init)

    def forward(self, x, patch_mask_dict=None):
        """
        [param] x: (n, t, d)
        """
        # separate cls-token
        x = x[:, 1:, :]
        
        # embed encoded patches
        x = self.decoder_embed(x)
        
        if self.process_type == 'indiv_frame':        
            # restore patch-mask
            if patch_mask_dict:
                mask_len = self.t_step * self.num_patches - x.shape[1]            
                mask_tokens = self.patch_mask_token.repeat(x.shape[0], mask_len, 1)            
                x = torch.cat([x, mask_tokens], dim=1)
                ids_restore = patch_mask_dict['patch_ids_restore']
                x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,x.shape[-1]))

            # pos_embed = 2d-spatial_embed + temporal_embed
            spatial_embed = self.spatial_embed.repeat(1, self.t_step, 1)
            temporal_embed = torch.repeat_interleave(self.temporal_embed, repeats=self.num_patches, dim=1)
            pos_embed = spatial_embed + temporal_embed
            x = x + pos_embed
        
        elif self.process_type == 'stack_frame':
            # restore patch-mask
            if patch_mask_dict:
                mask_len = self.num_patches - x.shape[1]
                mask_tokens = self.patch_mask_token.repeat(x.shape[0], mask_len, 1)            
                x = torch.cat([x, mask_tokens], dim=1)
                ids_restore = patch_mask_dict['patch_ids_restore']
                x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,x.shape[-1]))
            
            # pos_embed = 2d-spatial_embed 
            pos_embed = self.spatial_embed
            x = x + pos_embed
        
        # decoder
        x = self.emb_dropout(x)
        x, attn_maps = self.decoder(x)
        x = self.out_norm(x)   
        
        # predictor
        x = self.patch_pred(x)
        
        # info
        info = {}
        info['attn_maps'] = attn_maps
        
        return x, info
    