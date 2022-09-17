import numpy as np
import torch
import torch.nn as nn
from src.models.backbones.base import BaseBackbone
from einops import rearrange, repeat
from src.models.layers import Transformer
from src.common.train_utils import xavier_uniform_init
from src.common.vit_utils import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed
from src.common.vit_utils import get_1d_masked_input, get_3d_masked_input
    

class VIT(BaseBackbone):
    name = 'vit'
    def __init__(self,
                 obs_shape,
                 action_size,
                 process_type,
                 t_step,
                 patch_size,
                 pool,
                 enc_depth,
                 enc_dim, 
                 enc_mlp_dim,
                 enc_heads, 
                 emb_dropout,
                 dropout):

        super().__init__()
        t, c, i_h, i_w = obs_shape
        
        assert process_type in {'indiv_frame', 'stack_frame'}
        self.process_type = process_type
        if process_type == 'indiv_frame':
            self.in_channel = c
        elif process_type == 'stack_frame':
            self.in_channel = t * c
   
        p_h, p_w = patch_size
        assert i_h % p_h == 0 and i_w % p_w == 0, 'Image must be divisible by the patch size.'
        num_patches = (i_h // p_h) * (i_w // p_w)
        patch_dim = self.in_channel * p_h * p_w

        self.t_step = t_step
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.enc_dim = enc_dim
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, enc_dim))
        self.patch_embed = nn.Linear(patch_dim, enc_dim)

        self.spatial_embed = nn.Parameter(torch.randn(1, num_patches, enc_dim), requires_grad=False)
        self.temporal_embed = nn.Parameter(torch.randn(1, t_step, enc_dim), requires_grad=False)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.encoder = Transformer(dim=enc_dim, 
                                   depth=enc_depth, 
                                   heads=enc_heads, 
                                   mlp_dim=enc_mlp_dim, 
                                   dropout=dropout)
        self.out_norm = nn.LayerNorm(enc_dim)        
        self.act_pred = nn.Linear(enc_dim, action_size, bias=True)
        self.bc_pred = nn.Linear(enc_dim, action_size, bias=True)
        
        self._output_dim = enc_dim
        self._initialize_weights()
        
    def _initialize_weights(self):
        # initialize (and freeze) spatial pos_embed by 2d sin-cos embedding
        # initialize (and freeze) temporal pos_embed by 1d sin-cos embedding=
        
        T, P, D = self.t_step, self.num_patches, self.enc_dim
        
        enc_spatial_embed = get_2d_sincos_pos_embed(D, int((P)**.5))
        self.spatial_embed.copy_(torch.from_numpy(enc_spatial_embed).float().unsqueeze(0))
        self.spatial_embed.requires_grad = True

        enc_temporal_embed = get_1d_sincos_pos_embed_from_grid(D, np.arange(int(T)))
        self.temporal_embed.copy_(torch.from_numpy(enc_temporal_embed).float().unsqueeze(0))
        self.temporal_embed.requires_grad = True
        
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        
        # initialize nn.Linear and nn.LayerNorm
        self.apply(xavier_uniform_init)

    def forward(self, x, patch_mask_dict=None):
        """
        [param] x: (n, t, c, h, w)
        [param] patch_mask_dict
            patch_mask_type
            patch_mask
            patch_ids_keep
            patch_ids_restore
        [return] x: (n, t', d)
            proc_type == indiv_frame: t' = (t * num_patches) + 1
            proc_type == stack_frame: t' = (num_patches) + 1
        """
        ##########################
        # spatio-temporal MAE
        if self.process_type == 'indiv_frame':
            # patchify
            x = rearrange(x, 'n t c (h p1) (w p2) -> n (t h w) (p1 p2 c)', 
                          p1 = self.patch_size[0], p2 = self.patch_size[1])
            x = self.patch_embed(x)

            # pos_embed = 2d-spatial_embed + temporal_embed
            spatial_embed = self.spatial_embed.repeat(1, self.t_step, 1)
            temporal_embed = torch.repeat_interleave(self.temporal_embed, repeats=self.num_patches, dim=1)
            pos_embed = spatial_embed + temporal_embed
            x = x + pos_embed
            
            # masking: length -> length * mask_ratio
            if patch_mask_dict:
                ids_keep, mask_type = patch_mask_dict['patch_ids_keep'], patch_mask_dict['patch_mask_type']
                x = rearrange(x, 'n (t p) d -> n t p d', t = self.t_step, p = self.num_patches)
                x = get_3d_masked_input(x, ids_keep, mask_type)
            
        ##########################
        # image MAE
        elif self.process_type == 'stack_frame':
            # patchify
            x = rearrange(x, 'n t c (h p1) (w p2) -> n (h w) (t p1 p2 c)', 
                          p1 = self.patch_size[0], p2 = self.patch_size[1])
            x = self.patch_embed(x)

            # pos_embed = 2d-spatial_embed 
            pos_embed = self.spatial_embed
            x = x + pos_embed

            if patch_mask_dict:
                ids_keep, mask_type = patch_mask_dict['patch_ids_keep'], patch_mask_dict['patch_mask_type']
                x = rearrange(x, 'n p d -> n 1 p d')
                x = get_3d_masked_input(x, ids_keep, mask_type)

        # concatenate [cls] token
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([cls_token, x], dim=1)
            
        # apply Transformer blocks
        x = self.emb_dropout(x)
        x, attn_maps = self.encoder(x)
        x = self.out_norm(x)

        # extract infos
        info = {}
        info['attn_maps'] = attn_maps
        
        return x, info
    