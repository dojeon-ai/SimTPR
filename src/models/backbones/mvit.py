import torch.nn as nn
import torch
import numpy as np
from .base import BaseBackbone
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from src.common.train_utils import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed, get_1d_masked_input, get_3d_masked_input
from src.models.backbones.vit import Transformer


class MotionVIT(BaseBackbone):
    name = 'mvit'
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
        self.patch_size = patch_size
        self.num_patches = num_patches

        assert pool in {'mean'}, 'currently, pool must be mean (mean pooling)'

        ###########################################
        # Encoder 
        self.patch_embed = nn.Linear(patch_dim, enc_dim)

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
        # Patch Decoder
        self.decoder_embed = nn.Linear(enc_dim, dec_dim)        
        self.patch_mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))

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
        
        ########################################
        # Motion Decoder
        self.motion_decoder_embed = nn.Linear(enc_dim, dec_dim) 
        
        self.motion_mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.motion_dec_spatial_embed = nn.Parameter(torch.randn(1, num_patches, dec_dim), requires_grad=False) 
        self.motion_dec_temporal_embed = nn.Parameter(torch.randn(1, t_step, dec_dim), requires_grad=False)
        self.motion_dec_emb_dropout = nn.Dropout(emb_dropout)
        
        self.motion_decoder = Transformer(dim=dec_dim, 
                                          depth=dec_depth, 
                                          heads=dec_heads, 
                                          mlp_dim=dec_mlp_dim, 
                                          dropout=dropout)
        self.motion_dec_norm = nn.LayerNorm(dec_dim)        
        self.act_pred = nn.Linear(dec_dim, action_size, bias=True)
        
        self._output_dim = dec_dim
        
        self._initialize_weights()

    def _initialize_weights(self):
        # initialize (and freeze) spatial pos_embed by 2d sin-cos embedding
        # initialize (and freeze) temporal pos_embed by 1d sin-cos embedding
        enc_spatial_embed = get_2d_sincos_pos_embed(self.enc_spatial_embed.shape[-1], int((self.enc_spatial_embed.shape[1])**.5))
        self.enc_spatial_embed.copy_(torch.from_numpy(enc_spatial_embed).float().unsqueeze(0))
        self.enc_spatial_embed.requires_grad = True

        enc_temporal_embed = get_1d_sincos_pos_embed_from_grid(self.enc_temporal_embed.shape[-1], np.arange(int(self.enc_temporal_embed.shape[1])))
        self.enc_temporal_embed.copy_(torch.from_numpy(enc_temporal_embed).float().unsqueeze(0))
        self.enc_temporal_embed.requires_grad = True

        dec_spatial_embed = get_2d_sincos_pos_embed(self.dec_spatial_embed.shape[-1], int((self.dec_spatial_embed.shape[1])**.5))
        self.dec_spatial_embed.copy_(torch.from_numpy(dec_spatial_embed).float().unsqueeze(0))
        self.dec_spatial_embed.requires_grad = True

        dec_temporal_embed = get_1d_sincos_pos_embed_from_grid(self.dec_temporal_embed.shape[-1], np.arange(int(self.dec_temporal_embed.shape[1])))
        self.dec_temporal_embed.copy_(torch.from_numpy(dec_temporal_embed).float().unsqueeze(0))
        self.dec_temporal_embed.requires_grad = True
        
        motion_dec_spatial_embed = get_2d_sincos_pos_embed(self.motion_dec_spatial_embed.shape[-1], int((self.motion_dec_spatial_embed.shape[1])**.5))
        self.motion_dec_spatial_embed.copy_(torch.from_numpy(motion_dec_spatial_embed).float().unsqueeze(0))
        self.motion_dec_spatial_embed.requires_grad = True

        motion_dec_temporal_embed = get_1d_sincos_pos_embed_from_grid(self.motion_dec_temporal_embed.shape[-1], np.arange(int(self.motion_dec_temporal_embed.shape[1])))
        self.motion_dec_temporal_embed.copy_(torch.from_numpy(motion_dec_temporal_embed).float().unsqueeze(0))
        self.motion_dec_temporal_embed.requires_grad = True
        
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.patch_mask_token, std=.02)
        torch.nn.init.normal_(self.motion_mask_token, std=.02)

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
            
    def _get_decoder_attn_mask(self, done):
        N, T = done.shape
        
        # get uni-directional attn_mask (1: mask-out, 0: leave)
        L = self.t_step * (self.num_patches+1) # +1 for motion
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

    def forward(self, x):
        """
        [param] x: dict
            patch: (N, T * N_P, P_D) (T: t_step, N_P: num_patches)
            act: (N, T) 
            done: (N, T)
        """
            
        patch = x['patch']
        act = x['act']
        done = x['done']
        
        ##############################################
        # Encoder        
        x = self.patch_embed(patch)

        # add pos embed w/o act token
        # pos_embed = spatial_embed + temporal_embed
        enc_spatial_embed = self.enc_spatial_embed.repeat(1, self.t_step, 1)
        enc_temporal_embed = torch.repeat_interleave(self.enc_temporal_embed, repeats=self.num_patches, dim=1)
        enc_pos_embed = enc_spatial_embed + enc_temporal_embed
        x = x + enc_pos_embed

        # apply Transformer blocks
        x = self.emb_dropout(x)
        x = self.encoder(x)
        x = self.enc_norm(x)

        ##############################################
        # Motion Decoder

        # embed patches for motion decoder
        x = self.motion_decoder_embed(x)

        # pos-embed to patches
        dec_spatial_embed = self.motion_dec_spatial_embed.repeat(1, self.t_step, 1)
        dec_temporal_embed = torch.repeat_interleave(self.motion_dec_temporal_embed, repeats=self.num_patches, dim=1)
        dec_pos_embed = dec_spatial_embed + dec_temporal_embed
        x = x + dec_pos_embed

        # get motion tokens
        x_motion = self.motion_mask_token.repeat(x.shape[0], self.t_step, 1) 

        # pos-embed to motions
        x_motion = x_motion + self.dec_temporal_embed

        # concat patches with motions
        x = rearrange(x, 'n (t p) d -> n t p d', t = self.t_step, p = self.num_patches)
        x = torch.cat([x, x_motion.unsqueeze(2)], dim=2)
        x = rearrange(x, 'n t pa d -> n (t pa) d', t = self.t_step, pa = self.num_patches+1) # +1 for motion-mask
        
        # motion decoder
        x = self.motion_dec_emb_dropout(x)
        x = self.motion_decoder(x)
        x = self.motion_dec_norm(x)        
            
        return x
    
    
    def predict(self, x):
        N, D = x.shape[0], x.shape[-1]
        T, P = self.t_step, self.num_patches
        
        act_ids = (torch.arange(T, device=x.device)+1) * (P+1) - 1
        x_act = torch.gather(x, dim=1, index=act_ids.reshape(1,-1,1).repeat(N,1,D))
        
        # predict
        act_pred = self.act_pred(x_act)

        return act_pred