import numpy as np
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from torch.optim.lr_scheduler import _LRScheduler
from einops import rearrange


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    [param] embed_dim: dimension of the model
    [param] grid_size: int of the grid height and width
    [return] pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# 3d voxel masking
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
def get_random_1d_mask(shape, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.

    [param] shape: shape for batch of videos (N, S)
    [param] mask_ratio: portion of the mask.

    [return] ids_keep: ids to gather unmaksed patches from x.
    [return] mask: boolean mask to indicate the masked patches (N, S) (later used for loss computation)
    [return] ids_restore: x_mask can find its sequence via unshuffling with ids_restore. (used in decoder)
    """
    N, S = shape
    assert abs(S * (1-mask_ratio) - round(S * (1-mask_ratio))) < 1e-6, 'sequence length is not divisible to mask-ratio'
    len_keep = round(S * (1 - mask_ratio))

    # sample random noise
    noise = torch.rand(N, S)  # noise in [0, 1]
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, S])
    mask[:, :len_keep] = 0

    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return ids_keep, mask, ids_restore


def get_random_3d_mask(shape, mask_ratio, mask_type):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.

    [param] shape: shape for batch of videos (N, T, P) (T: time_steps, P: num_patches).
    [param] mask_ratio: portion of the mask.
    [param] mask_type: masking option which is one of ['agnostic', 'space', 'time'].

    [return] ids_keep: ids to gather unmaksed patches from x.
    [return] mask: boolean mask to indicate the masked patches (N, T*P) (later used for loss computation)
    [return] ids_restore: x_mask can find its sequence via unshuffling with ids_restore. (used in decoder)
    """
    assert mask_type in {'agnostic', 'space', 'time'}, 'mask_type should be defined within [agnostic, space, time]'

    # get random noise for masking
    if mask_type == 'agnostic':
        ids_keep, mask, ids_restore = random_3d_agnostic_masking(shape, mask_ratio)
            
    # space masking is identical to the tube masking in Video MAE
    # Video MAE: https://arxiv.org/abs/2203.12602
    elif mask_type == 'space': 
        ids_keep, mask, ids_restore = random_3d_space_masking(shape, mask_ratio)

    elif mask_type == 'time':
        ids_keep, mask, ids_restore = random_3d_time_masking(shape, mask_ratio)

    return ids_keep, mask, ids_restore


def random_3d_agnostic_masking(shape, mask_ratio):
    N, T, P = shape
    L = T*P
    assert abs(L * (1-mask_ratio) - round(L * (1-mask_ratio))) < 1e-6, 'sequence length is not divisible to mask-ratio'
    len_keep = round(L * (1 - mask_ratio))

    # sample random noise
    noise = torch.rand(N, L)  # noise in [0, 1]
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L])
    mask[:, :len_keep] = 0

    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return ids_keep, mask, ids_restore


def random_3d_space_masking(shape, mask_ratio):
    N, T, P = shape
    assert abs(P * (1-mask_ratio) - round(P * (1-mask_ratio))) < 1e-6, 'num-pathces are not divisible to mask-ratio'
    len_keep = round(P * (1 - mask_ratio))

    # sample random noise
    noise = torch.rand(N, P)  # noise in [0, 1]
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, P])
    mask[:, :len_keep] = 0

    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    # repeat temporally
    mask = mask.repeat(1, T)

    # get temporally extended restoring index
    temporal_ids = torch.arange(T*P).reshape(-1,T,len_keep)
    temporal_ids = rearrange(temporal_ids, 'n t l_k -> t (n l_k)')

    ids_restore = torch.index_select(temporal_ids, dim=-1, index=ids_restore.flatten()) #.squeeze(0)
    ids_restore = torch.vstack(torch.hsplit(ids_restore, N)).reshape(N, -1)

    return ids_keep, mask, ids_restore


def random_3d_cube_masking(shape, mask_ratio):
    #import pdb; pdb.set_trace()
    N, T, P = shape #32 * 16 *49
    patch_t = 2
    repeats_time = T // patch_t
    #assert abs(L * (1-mask_ratio) - round(L * (1-mask_ratio))) < 1e-6, 'sequence length is not divisible to mask-ratio'
    # import pdb; pdb.set_trace()
    len_keep = round(P * (1 - mask_ratio)) # 25

    # sample random noise
    noise = torch.rand(N, patch_t, P)  # noise in [0, 1]
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=-1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=-1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :, :len_keep]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, patch_t, P])
    mask[:, :, :len_keep] = 0

    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=-1, index=ids_restore)

    # repeat temporally
    mask = mask.repeat_interleave(repeats_time, dim= 1).reshape(mask.size(0), -1)

    spatial_indexing = torch.arange(T).reshape(-1, 1) * P
    spatial_ids = torch.repeat_interleave(ids_restore, repeats=repeats_time, dim=1)

    temporal_ids = torch.arange(T*P).reshape(-1,T,len_keep)
    temporal_ids = rearrange(temporal_ids, 'n t l_k -> t (n l_k)').repeat(N, 1, 1)

    ids_restore = torch.stack([ torch.stack([torch.index_select(temporal_ids[i,j], dim=-1, index= spatial_ids[i,j]) for j in range(len(spatial_ids[i]))]).reshape(-1) \
                    for i in range(len(spatial_ids))])

    # get temporally extended restoring index
    ids_keep = torch.repeat_interleave(ids_keep, repeats=repeats_time, dim=1) + spatial_indexing
    ids_keep = ids_keep.reshape(ids_keep.size(0), -1)

    return ids_keep, mask, ids_restore


def get_random_3d_mask(shape, mask_ratio, mask_type):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.

    [param] shape: shape for batch of videos (N, T, P) (T: time_steps, P: num_patches).
    [param] mask_ratio: portion of the mask.
    [param] mask_type: masking option which is one of ['agnostic', 'space', 'time'].

    [return] ids_keep: ids to gather unmaksed patches from x.
    [return] mask: boolean mask to indicate the masked patches (N, T*P) (later used for loss computation)
    [return] ids_restore: x_mask can find its sequence via unshuffling with ids_restore. (used in decoder)
    """
    assert mask_type in {'agnostic', 'space', 'time', 'cube'}, 'mask_type should be defined within [agnostic, space, time]'

    # get random noise for masking
    if mask_type == 'agnostic':
        ids_keep, mask, ids_restore = random_3d_agnostic_masking(shape, mask_ratio)
            
    # space masking is identical to the tube masking in Video MAE
    # Video MAE: https://arxiv.org/abs/2203.12602
    elif mask_type == 'space': 
        ids_keep, mask, ids_restore = random_3d_space_masking(shape, mask_ratio)

    elif mask_type == 'time':
        ids_keep, mask, ids_restore = random_3d_time_masking(shape, mask_ratio)

    elif mask_type == 'cube':
        ids_keep, mask, ids_restore = random_3d_cube_masking(shape, mask_ratio)

    return ids_keep, mask, ids_restore


def get_1d_masked_input(x, ids_keep):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.

    [param] x: batch of videos with a shape (N, S, D).
    [param] ids_keep: ids to gather unmaksed patches from x.

    [return] x_masked: unmasked patches of x with a shape (N, L, D) (L = (1-mask_ratio) * S)
    """  
    N, S, D = x.shape
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    return x_masked


def get_3d_masked_input(x, ids_keep, mask_type):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.

    [param] x: batch of videos with a shape (N, T, P, D) (T: time_steps, P: num_patches).
    [param] ids_keep: ids to gather unmaksed patches from x.
    [param] mask_type: masking option which is one of ['agnostic', 'space', 'time'].

    [return] x_masked: unmasked patches of x with a shape (N, L, D) (L = (1-mask_ratio) * T * P)
    """  
    assert mask_type in {'agnostic', 'space', 'time', 'cube'}, 'mask_type should be defined within [agnostic, space, time]'

    N, T, P, D = x.shape

    if mask_type == 'agnostic':    
        x = rearrange(x, 'n t p d -> n (t p) d')
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
            
    elif mask_type == 'space': 
        x_masked = torch.gather(x, dim=2, index=ids_keep.reshape(N,1,-1,1).repeat(1, T, 1, D))
        x_masked = rearrange(x_masked, 'n t l_k d -> n (t l_k) d') 

    elif mask_type == 'time':
        x_masked = torch.gather(x, dim=1, index=ids_keep.reshape(N,-1,1,1).repeat(1, 1, P, D))
        x_masked = rearrange(x_masked, 'n l_k t d -> n (l_k t) d') 

    elif mask_type == 'cube':
        x = rearrange(x, 'n t p d -> n (t p) d')
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    return x_masked


def restore_masked_input(x_masked, ids_restore, mask_tokens=None):
    """
    [param] x_masked: (n, mT, d)
    [param] ids_restore: (n, T)
    [param] mask_tokens: (n, T-mT, d)
    [return] x: (n, T, d)
    """
    # mT = t * n_p * mask_ratio
    # T = t * n_p
    n, mT, d = x_masked.shape
    _, T = ids_restore.shape
    
    if mask_tokens is None:
        mask_tokens = torch.zeros((n, T-mT, d), device=x_masked.device)
    
    x = torch.cat([x_masked, mask_tokens], dim=1)    
    x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,d)) 
    
    return x



if __name__ == '__main__':
    print('[TEST Mask Functions]')
    device = torch.device('cuda:0')

    #-------------------------------------------------------------
    print('[1. TEST batch_size=1]')
    N, T, P, D = 1, 10, 5, 3
    x = torch.randn((N, T, P, D)).to(device)
    shape = (N, T, P)
    mask_ratio = 0.8
    mask_tokens = torch.zeros((N,int(T*P*mask_ratio),D)).to(device)

    def test_masking(x, mask_ratio, mask_type):
        #import pdb; pdb.set_trace()
        
        ids_keep, mask, ids_restore = get_random_3d_mask(shape, mask_ratio, mask_type)
        import pdb; pdb.set_trace()
        x_masked = get_3d_masked_input(x, ids_keep.to(device), mask_type)
        x_ = torch.cat([x_masked, mask_tokens], dim=1)
        
        x_= torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,D).to(device)) # unshuffle

        num_masked = torch.sum(torch.sum(x_ - x.reshape(N, T*P, D), -1) == 0)
        assert num_masked == round(N*T*P*(1-mask_ratio))
        print('[passed]')
    
    print('[1.1. TEST agnostic masking]')
    mask_type='agnostic'
    test_masking(x, mask_ratio, mask_type)

    print('[1.2. TEST space-only masking]')
    mask_type='space'
    test_masking(x, mask_ratio, mask_type)

    print('[1.3. TEST time-only masking]')
    mask_type='time'
    test_masking(x, mask_ratio, mask_type)

    print('[1.4. TEST cube masking]')
    mask_type='cube'
    test_masking(x, mask_ratio, mask_type)

    #-------------------------------------------------------------

    #-------------------------------------------------------------
    print('[2. TEST larger batch size]')
    N, T, P, D = 4, 5, 10, 3
    x = torch.randn((N, T, P, D)).to(device)
    shape = (N, T, P)
    mask_ratio = 0.8
    mask_tokens = torch.zeros((N,int(T*P*mask_ratio),D)).to(device)

    print('[2.1. TEST agnostic masking]')
    mask_type='agnostic'
    test_masking(x, mask_ratio, mask_type)

    print('[2.2. TEST space-only masking]')
    mask_type='space'
    test_masking(x, mask_ratio, mask_type)

    print('[2.3. TEST time-only masking]')
    mask_type='time'
    test_masking(x, mask_ratio, mask_type)

    # print('[2.4. TEST cube masking]') # Can't test this as 5 % 2 == 1
    # mask_type='cube'
    # test_masking(x, mask_ratio, mask_type)

    #-------------------------------------------------------------

    #-------------------------------------------------------------
    print('[3. TEST different masking ratio]')
    N, T, P, D = 4, 6, 10, 3
    x = torch.randn((N, T, P, D)).to(device)
    shape = (N, T, P)
    mask_ratio = 0.5
    mask_tokens = torch.zeros((N,int(T*P*mask_ratio),D)).to(device)

    # print('[3.1. TEST agnostic masking]')
    # mask_type='agnostic'
    # test_masking(x, mask_ratio, mask_type)

    # print('[3.2. TEST space-only masking]')
    # mask_type='space'
    # test_masking(x, mask_ratio, mask_type)

    # print('[3.3. TEST time-only masking]')
    # mask_type='time'
    # test_masking(x, mask_ratio, mask_type)

    print('[3.4. TEST cube masking]')
    mask_type='cube'
    test_masking(x, mask_ratio, mask_type)

    #-------------------------------------------------------------