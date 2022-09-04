import torch.nn as nn
import torch
import numpy as np
from einops import rearrange, repeat
from .base import BaseHead
from src.models.layers import Transformer
from src.common.train_utils import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed
from src.common.train_utils import get_1d_masked_input, get_3d_masked_input


class VITHead(BaseHead):
    name = 'vit_head'
    def __init__(self, 
                 obs_shape,
                 action_size,
                 patch_size,
                 t_step,
                 enc_dim, 
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
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.dec_dim = dec_dim
        
        self.decoder_embed = nn.Linear(enc_dim, dec_dim)                
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
        self.spatial_embed.requires_grad = True

        dec_temporal_embed = get_1d_sincos_pos_embed_from_grid(D, np.arange(int(T)))
        self.temporal_embed.copy_(torch.from_numpy(dec_temporal_embed).float().unsqueeze(0))
        self.temporal_embed.requires_grad = True

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.patch_mask_token, std=.02)
        
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

    def forward(self, x, patch_mask_dict=None):
        """
        [param] x: (N, T*(P+1), D)
        """
        # separate cls-token
        x = x[:, self.t_step:, :]
        
        # embed encoded patches
        x = self.decoder_embed(x)
        
        # restore patch-mask
        if patch_mask_dict:
            mask_len = self.t_step * self.num_patches - x.shape[1]            
            mask_tokens = self.patch_mask_token.repeat(x.shape[0], mask_len, 1)            
            x = torch.cat([x, mask_tokens], dim=1)
            ids_restore = patch_mask_dict['patch_ids_restore']
            x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,x.shape[-1]))
        
        # pos-embed to patches
        spatial_embed = self.spatial_embed.repeat(1, self.t_step, 1)
        temporal_embed = torch.repeat_interleave(self.temporal_embed, repeats=self.num_patches, dim=1)
        pos_embed = spatial_embed + temporal_embed
        x = x + pos_embed
        
        # decoder
        x = self.emb_dropout(x)
        x, _ = self.decoder(x)
        x = self.out_norm(x)   
        
        # predictor
        x = self.patch_pred(x)
        
        return x
    
