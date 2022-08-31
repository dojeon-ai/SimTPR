import torch.nn as nn
import torch
import numpy as np
from .base import BaseBackbone
from einops import rearrange, repeat
from src.models.layers import Transformer
from src.common.train_utils import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed
from src.common.train_utils import get_1d_masked_input, get_3d_masked_input
    

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
                 emb_dropout,
                 dropout,
                 renormalize):

        super().__init__()
        # TODO: error when frame is not 1.
        # frame, channel, image_height, image_width = obs_shape
        # image_channel = frame * channel
        frame, channel, image_height, image_width = obs_shape
        image_channel = channel        
        patch_height, patch_width = patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = image_channel * patch_height * patch_width

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
        
        assert pool in {'identity', 'cls_concat', 'cls_last'}
        self.pool = pool
        self.renormalize = renormalize
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
    
    def forward(self, x, patch_mask_dict=None, get_attn_map=False):
        """
        [param] x: (N, T, C, H, W)
        [param] patch_mask_dict
            patch_mask_type
            patch_mask
            patch_ids_keep
            patch_ids_restore
        [param] act_mask_dict
            act_mask
            act_ids_keep
            act_ids_restore
        [return] x: (N, T*(P+1), D) (+1 for cls)
        """
        if len(x.shape) == 4:
            x = rearrange(x, 'n t (h p1) (w p2) -> n (t h w) (p1 p2)', 
                          p1 = self.patch_size[0], p2 = self.patch_size[1])

        elif len(x.shape) == 5:
            x = rearrange(x, 'n t c (h p1) (w p2) -> n (t h w) (p1 p2 c)', 
                          p1 = self.patch_size[0], p2 = self.patch_size[1])
        
        # patch embed
        x = self.patch_embed(x)

        # add pos embed w/o act token
        # pos_embed = spatial_embed + temporal_embed
        spatial_embed = self.spatial_embed.repeat(1, self.t_step, 1)
        temporal_embed = torch.repeat_interleave(self.temporal_embed, repeats=self.num_patches, dim=1)
        pos_embed = spatial_embed + temporal_embed
        x = x + pos_embed
        
        # masking: length -> length * mask_ratio
        if patch_mask_dict:
            ids_keep, mask_type = patch_mask_dict['patch_ids_keep'], patch_mask_dict['patch_mask_type']
            x = rearrange(x, 'n (t p) d -> n t p d', t = self.t_step, p = self.num_patches)
            x = get_3d_masked_input(x, ids_keep, mask_type)
            
        # concatenate [cls] token
        cls_tokens = self.cls_token.repeat(x.shape[0], self.t_step, 1)      
        cls_tokens = cls_tokens + self.temporal_embed
        x = torch.cat([cls_tokens, x], dim=1)
            
        # apply Transformer blocks
        x = self.emb_dropout(x)
        x, attn_maps = self.encoder(x)
        x = self.out_norm(x)

        # pooling
        if self.pool == 'identity':
            x = x
            
        elif self.pool == 'cls_concat':
            x = x[:,:self.t_step,:]
            x = rearrange(x, 'n t d -> n (t d)')
            
        elif self.pool == 'cls_last':
            x = x[:,self.t_step-1:self.t_step,:]
            x = rearrange(x, 'n t d -> n (t d)')
            
        if self.renormalize:
            x = self._renormalize(x)
        
        if get_attn_map:
            return x, attn_maps
        else:
            return x

    def predict_act(self, x):
        """
        [param] x: (N, T*(P+1), D) (+1 for cls)
        """
        idm_act = self.act_pred(x[:, :self.t_step-1, :])
        bc_act = self.bc_pred(x[:, self.t_step-1:self.t_step, :])
        x = torch.cat([idm_act, bc_act], dim=1)
        
        return x
        
    