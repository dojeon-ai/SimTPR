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
        self.act_embed = nn.Embedding(action_size, dec_dim)
        
        self.patch_mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.act_mask_token = nn.Parameter(torch.zeros(1, 1, enc_dim))

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
        self.act_pred = nn.Linear(dec_dim, action_size, bias=True)
        
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
        torch.nn.init.normal_(self.act_mask_token, std=.02)
        
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

    def project(self, x, act, done, use_action, patch_mask_dict, act_mask_dict):
        """
        [param] x: (N, T*(P+1), D)
        """
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
        
        # concatenate action if needed
        if use_action:
            x_act = self.act_embed(act)
            
            # mask & restore actions
            if act_mask_dict:
                x_act = get_1d_masked_input(x_act, act_mask_dict['act_ids_keep'])            
                act_mask_len = (self.t_step - 1) - x_act.shape[1]
                act_mask_tokens = self.act_mask_token.repeat(x.shape[0], act_mask_len, 1) 

                x_act = torch.cat([x_act, act_mask_tokens], dim=1)
                x_act = torch.gather(x_act, dim=1, index=act_mask_dict['act_ids_restore'].unsqueeze(-1).repeat(1,1,x_act.shape[-1]))
                
            # pos-embed to actions
            x_act = x_act + self.temporal_embed[:, :-1, :]
            
        else:
            x_act = torch.zeros((x.shape[0], self.t_step-1, x.shape[-1]), device=x.device)
        
        # mask-out done-actions
        x_act = x_act * (1-done.unsqueeze(-1))
        
        # concat patches with actions
        x = torch.cat([x, x_act], dim=1)
        
        # decoder
        x = self.emb_dropout(x)
        x = self.decoder(x)
        x = self.out_norm(x)    
        
        return x
    
    def predict(self, x):
        L = self.t_step * self.num_patches
        patch_pred = self.patch_pred(x[:, :L, :])
        act_pred = self.act_pred(x[:, L:, :])

        return patch_pred, act_pred

    def forward(self, x, act, done, use_action=False, patch_mask_dict=None, act_mask_dict=None):
        x = self.project(x, act, done, use_action, patch_mask_dict, act_mask_dict)
        patch_pred, act_pred = self.predict(x)
        
        return patch_pred, act_pred