import torch.nn as nn
import torch
import numpy as np
from .base import BaseBackbone
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from src.models.layers import Transformer
from src.common.train_utils import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed, get_1d_masked_input, get_3d_masked_input
    

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
    
    
class VITEncoder(nn.Module):
    def __init__(self,
                 obs_shape,
                 action_size,
                 patch_size,
                 num_patches,
                 patch_dim,
                 t_step,
                 pool,
                 enc_depth,
                 enc_dim, 
                 enc_mlp_dim,
                 enc_heads,
                 emb_dropout,
                 dropout):
        
        super().__init__()
        self.num_patches = num_patches
        self.t_step = t_step
        
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

    def forward(self, patch, done, patch_mask_dict):
        # patch embed
        x = self.patch_embed(patch)

        # add pos embed w/o act token
        # pos_embed = spatial_embed + temporal_embed
        spatial_embed = self.spatial_embed.repeat(1, self.t_step, 1)
        temporal_embed = torch.repeat_interleave(self.temporal_embed, repeats=self.num_patches, dim=1)
        pos_embed = spatial_embed + temporal_embed
        x = x + pos_embed

        # get attention mask (action is always not used in encoder)
        attn_mask = get_attn_mask(self.t_step, self.num_patches, done, use_action=False)
        attn_mask = attn_mask[:, :-(self.t_step-1), :-(self.t_step-1)]
        
        # masking: length -> length * mask_ratio
        if patch_mask_dict:
            ids_keep, mask_type = patch_mask_dict['patch_ids_keep'], patch_mask_dict['patch_mask_type']
            x = rearrange(x, 'n (t p) d -> n t p d', t = self.t_step, p = self.num_patches)
            x = get_3d_masked_input(x, ids_keep, mask_type)
            
            attn_mask = torch.gather(attn_mask, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, attn_mask.shape[-1]))
            attn_mask = torch.gather(attn_mask, dim=2, index=ids_keep.unsqueeze(1).repeat(1, attn_mask.shape[1], 1))
            
        # apply Transformer blocks
        x = self.emb_dropout(x)
        x = self.encoder(x, attn_mask)
        x = self.out_norm(x)
        
        return x
    
    
class VITDecoder(nn.Module):
    def __init__(self,
                 obs_shape,
                 action_size,
                 patch_size,
                 num_patches,
                 patch_dim,
                 t_step,
                 pool,
                 enc_dim,
                 dec_depth, 
                 dec_dim, 
                 dec_mlp_dim,
                 dec_heads, 
                 emb_dropout,
                 dropout):
        
        super().__init__()
        self.num_patches = num_patches
        self.t_step = t_step
        
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

    def forward(self, x, act, done, use_action, patch_mask_dict, act_mask_dict):        
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
        
        # concat patches with actions
        x = torch.cat([x, x_act], dim=1)
        
        # casual attention mask
        attn_mask = get_attn_mask(self.t_step, self.num_patches, done, use_action)
        
        # decoder
        x = self.emb_dropout(x)
        x = self.decoder(x, attn_mask)
        x = self.out_norm(x)    
        
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
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        
        assert pool in {'mean'}, 'currently, pool must be mean (mean pooling)'

        self.encoder = VITEncoder(obs_shape,
                                  action_size,
                                  patch_size,
                                  num_patches,
                                  patch_dim,
                                  t_step,
                                  pool,
                                  enc_depth,
                                  enc_dim, 
                                  enc_mlp_dim,
                                  enc_heads,
                                  emb_dropout,
                                  dropout)
        
        self.decoder = VITDecoder(obs_shape,
                                  action_size,
                                  patch_size,
                                  num_patches,
                                  patch_dim,
                                  t_step,
                                  pool,
                                  enc_dim,
                                  dec_depth, 
                                  dec_dim, 
                                  dec_mlp_dim,
                                  dec_heads, 
                                  emb_dropout,
                                  dropout)
        
        self.patch_pred = nn.Linear(dec_dim, patch_dim, bias=True)
        self.act_pred = nn.Linear(dec_dim, action_size, bias=True)

        self._output_dim = dec_dim
        self._initialize_weights()

        
    def _initialize_weights(self):
        # initialize (and freeze) spatial pos_embed by 2d sin-cos embedding
        # initialize (and freeze) temporal pos_embed by 1d sin-cos embedding=
        
        T, P = self.t_step, self.num_patches
        E_D, D_D = self.enc_dim, self.dec_dim
        
        enc_spatial_embed = get_2d_sincos_pos_embed(E_D, int((P)**.5))
        self.encoder.spatial_embed.copy_(torch.from_numpy(enc_spatial_embed).float().unsqueeze(0))
        self.encoder.spatial_embed.requires_grad = True

        enc_temporal_embed = get_1d_sincos_pos_embed_from_grid(E_D, np.arange(int(T)))
        self.encoder.temporal_embed.copy_(torch.from_numpy(enc_temporal_embed).float().unsqueeze(0))
        self.encoder.temporal_embed.requires_grad = True

        dec_spatial_embed = get_2d_sincos_pos_embed(D_D, int((P)**.5))
        self.decoder.spatial_embed.copy_(torch.from_numpy(dec_spatial_embed).float().unsqueeze(0))
        self.decoder.spatial_embed.requires_grad = True

        dec_temporal_embed = get_1d_sincos_pos_embed_from_grid(D_D, np.arange(int(T)))
        self.decoder.temporal_embed.copy_(torch.from_numpy(dec_temporal_embed).float().unsqueeze(0))
        self.decoder.temporal_embed.requires_grad = True

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.decoder.patch_mask_token, std=.02)
        torch.nn.init.normal_(self.decoder.act_mask_token, std=.02)
        
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
            
    
    def forward(self, x, use_action=False, patch_mask_dict=None, act_mask_dict=None):
        """
        [param] x: dict
            patch: (N, L, D) (L = T * P, T: t_step, P: num_patches)
            act: (N, T-1)
            done: (N, T-1)
        [param] patch_mask_dict
            patch_mask_type
            patch_mask
            patch_ids_keep
            patch_ids_restore
        [param] act_mask_dict
            act_mask
            act_ids_keep
            act_ids_restore
        [return] x: (N, L, D) (if use_action: L=T*(P+1)-1 else: L=T*P)
        """
        patch = x['patch']
        act = x['act']
        done = x['done']
        
        # encoder (N, L, D)
        x = self.encoder(patch, done, patch_mask_dict)
        
        # decoder (N, L+T-1, D)
        # actions are concatenated at the last
        x = self.decoder(x, act, done, use_action, patch_mask_dict, act_mask_dict)  
        
        # predictor
        L = self.t_step * self.num_patches
        patch_pred = self.patch_pred(x[:, :L, :])
        act_pred = self.act_pred(x[:, L:, :])
        
        return patch_pred, act_pred
    