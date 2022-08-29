import numpy as np
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from einops import rearrange


def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, act_token=False):
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
    if act_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
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
    omega = np.arange(embed_dim // 2, dtype=np.float)
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
def random_3d_masking(x, mask_ratio, mask_type):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.

    [param] x: batch of videos with a shape (N, T, P, D) (T: time_steps, P: num_patches).
    [param] mask_ratio: portion of the mask.
    [param] mask_type: masking option which is one of ['agnostic', 'space', 'time'].

    [return] x_masked: flattend view of the un-masked patches (N, T*P*mask_ratio, D).
    [return] mask: boolean mask to indicate the masked patches (N, T*P) (later used for loss computation)
    [return] ids_restore: x_mask can find its sequence via unshuffling with ids_restore. (used in decoder)
    """
    assert mask_type in {'agnostic', 'space', 'time'}, 'mask_type should be defined within [agnostic, space, time]'

    # get random noise for masking
    if mask_type == 'agnostic':
        x_masked, mask, ids_restore = random_3d_agnostic_masking(x, mask_ratio)
            
    # space masking is identical to the tube masking in Video MAE
    # Video MAE: https://arxiv.org/abs/2203.12602
    elif mask_type == 'space': 
        x_masked, mask, ids_restore = random_3d_space_masking(x, mask_ratio)

    elif mask_type == 'time':
        x_masked, mask, ids_restore = random_3d_time_masking(x, mask_ratio)

    return x_masked, mask, ids_restore


def random_3d_agnostic_masking(x, mask_ratio):
    N, T, P, D = x.shape  # batch, time_step, spatial_size, dim
    L = T*P
    assert (L * (1-mask_ratio) - round(L * (1-mask_ratio))) < 1e-6, 'sequence length is not divisible to mask-ratio'
    len_keep = round(L * (1 - mask_ratio))

    # sample random noise
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]

    # randomly mask out 3d-voxels
    x = rearrange(x, 'n t p d -> n (t p) d')
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0

    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


def random_3d_space_masking(x, mask_ratio):
    N, T, P, D = x.shape  # batch, time_step, spatial_size, dim
    assert (P * (1-mask_ratio) - round(P * (1-mask_ratio))) < 1e-6, 'num-pathces are not divisible to mask-ratio'
    len_keep = round(P * (1 - mask_ratio))

    # sample random noise
    noise = torch.rand(N, P, device=x.device)  # noise in [0, 1]
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]

    # randomly mask out 3d-voxels
    x_masked = torch.gather(x, dim=2, index=ids_keep.reshape(N,1,-1,1).repeat(1, T, 1, D))
    x_masked = rearrange(x_masked, 'n t l_k d -> n (t l_k) d') 

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, P], device=x.device)
    mask[:, :len_keep] = 0

    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    # repeat temporally
    mask = mask.repeat(1, T)

    # get temporally extended restoring index
    temporal_ids = torch.arange(T*P, device=x.device).reshape(-1,T,len_keep)
    temporal_ids = rearrange(temporal_ids, 'n t l_k -> t (n l_k)')

    ids_restore = torch.index_select(temporal_ids, dim=-1, index=ids_restore.flatten()) #.squeeze(0)
    ids_restore = torch.vstack(torch.hsplit(ids_restore, N)).reshape(N, -1)

    return x_masked, mask, ids_restore


def random_3d_time_masking(x, mask_ratio):
    N, T, P, D = x.shape  # batch, time_step, spatial_size, dim
    assert (T * (1-mask_ratio) - round(T * (1-mask_ratio))) < 1e-6, 'time-steps are not divisible to mask-ratio'
    len_keep = round(T * (1 - mask_ratio))

    # sample random noise
    noise = torch.rand(N, T, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]

    # randomly mask out 3d-voxels
    x_masked = torch.gather(x, dim=1, index=ids_keep.reshape(N,-1,1,1).repeat(1, 1, P, D))
    x_masked = rearrange(x_masked, 'n l_k t d -> n (l_k t) d') 

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, T], device=x.device)
    mask[:, :len_keep] = 0

    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    # repeat spatially
    mask = torch.repeat_interleave(mask, repeats=P, dim=1)

    # get spatially extended restoring index
    spatial_ids = torch.repeat_interleave(ids_restore, repeats=P, dim=1) * P
    ids_restore = spatial_ids + torch.arange(P, device=x.device).reshape(1,-1).repeat(N, T)

    return x_masked, mask, ids_restore


class LinearScheduler(object):
    def __init__(self, initial_value, final_value, step_size):
        """
        Linear Interpolation between initial_value to the final_value
        [params] initial_value (float) initial output value
        [params] final_value (float) final output value
        [params] step_size (int) number of timesteps to lineary anneal initial value to the final value
        """
        self.initial_value = initial_value
        self.final_value   = final_value
        self.step_size = step_size
        
    def get_value(self, step):
        """
        Return the scheduled value
        """
        interval = (self.initial_value - self.final_value) / self.step_size
        # After the schedule_timesteps, final value is returned
        if self.final_value < self.initial_value:
            return max(self.initial_value - interval * step, self.final_value)
        else:
            return min(self.initial_value - interval * step, self.final_value)


class RMS(object):
    """running mean and std """
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs +
                 torch.square(delta) * self.n * bs /
                 (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S


if __name__ == '__main__':
    print('[TEST Mask Functions]')
    device = torch.device('cuda:0')

    #-------------------------------------------------------------
    print('[1. TEST batch_size=1]')
    N, T, P, D = 1, 10, 5, 3
    x = torch.randn((N, T, P, D)).to(device)
    mask_ratio = 0.8
    mask_tokens = torch.zeros((N,int(T*P*mask_ratio),D)).to(device)

    print('[1.1. TEST agnostic masking]')
    x_masked, mask, ids_restore = random_3d_masking(x, mask_ratio, mask_type='agnostic')
    x_ = torch.cat([x_masked, mask_tokens], dim=1)
    x_= torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,D)) # unshuffle

    num_masked = torch.sum(torch.sum(x_ - x.reshape(N, T*P, D), -1) == 0)
    assert num_masked == round(N*T*P*(1-mask_ratio))
    print('[passed]')

    print('[1.2. TEST space-only masking]')
    x_masked, mask, ids_restore = random_3d_masking(x, mask_ratio, mask_type='space')
    x_ = torch.cat([x_masked, mask_tokens], dim=1)
    x_= torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,D)) # unshuffle

    num_masked = torch.sum(torch.sum(x_ - x.reshape(N, T*P, D), -1) == 0)
    assert num_masked == round(N*T*P*(1-mask_ratio))
    print('[passed]')

    print('[1.3. TEST time-only masking]')
    x_masked, mask, ids_restore = random_3d_masking(x, mask_ratio, mask_type='time')
    x_ = torch.cat([x_masked, mask_tokens], dim=1)

    x_= torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,D)) # unshuffle
    num_masked = torch.sum(torch.sum(x_ - x.reshape(N, T*P, D), -1) == 0)
    assert num_masked == round(N*T*P*(1-mask_ratio))
    print('[passed]')

    #-------------------------------------------------------------

    #-------------------------------------------------------------
    print('[2. TEST larger batch size]')
    N, T, P, D = 4, 5, 10, 3
    x = torch.randn((N, T, P, D)).to(device)
    mask_ratio = 0.8
    mask_tokens = torch.zeros((N,int(T*P*mask_ratio),D)).to(device)

    print('[2.1. TEST agnostic masking]')
    x_masked, mask, ids_restore = random_3d_masking(x, mask_ratio, mask_type='agnostic')
    x_ = torch.cat([x_masked, mask_tokens], dim=1)
    x_= torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,D)) # unshuffle

    num_masked = torch.sum(torch.sum(x_ - x.reshape(N, T*P, D), -1) == 0)
    assert num_masked == round(N*T*P*(1-mask_ratio))
    print('[passed]')

    print('[2.2. TEST space-only masking]')
    x_masked, mask, ids_restore = random_3d_masking(x, mask_ratio, mask_type='space')
    x_ = torch.cat([x_masked, mask_tokens], dim=1)
    x_= torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,D)) # unshuffle

    num_masked = torch.sum(torch.sum(x_ - x.reshape(N, T*P, D), -1) == 0)
    assert num_masked == round(N*T*P*(1-mask_ratio))
    print('[passed]')

    print('[2.3. TEST time-only masking]')
    x_masked, mask, ids_restore = random_3d_masking(x, mask_ratio, mask_type='time')
    x_ = torch.cat([x_masked, mask_tokens], dim=1)

    x_= torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,D)) # unshuffle
    num_masked = torch.sum(torch.sum(x_ - x.reshape(N, T*P, D), -1) == 0)
    assert num_masked == round(N*T*P*(1-mask_ratio))
    print('[passed]')

    #-------------------------------------------------------------

    #-------------------------------------------------------------
    print('[3. TEST different masking ratio]')
    N, T, P, D = 4, 6, 10, 3
    x = torch.randn((N, T, P, D)).to(device)
    mask_ratio = 0.5
    mask_tokens = torch.zeros((N,int(T*P*mask_ratio),D)).to(device)

    print('[3.1. TEST agnostic masking]')
    x_masked, mask, ids_restore = random_3d_masking(x, mask_ratio, mask_type='agnostic')
    x_ = torch.cat([x_masked, mask_tokens], dim=1)
    x_= torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,D)) # unshuffle

    num_masked = torch.sum(torch.sum(x_ - x.reshape(N, T*P, D), -1) == 0)
    assert num_masked == round(N*T*P*(1-mask_ratio))
    print('[passed]')

    print('[3.2. TEST space-only masking]')
    x_masked, mask, ids_restore = random_3d_masking(x, mask_ratio, mask_type='space')
    x_ = torch.cat([x_masked, mask_tokens], dim=1)
    x_= torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,D)) # unshuffle

    num_masked = torch.sum(torch.sum(x_ - x.reshape(N, T*P, D), -1) == 0)
    assert num_masked == round(N*T*P*(1-mask_ratio))
    print('[passed]')

    print('[3.3. TEST time-only masking]')
    x_masked, mask, ids_restore = random_3d_masking(x, mask_ratio, mask_type='time')
    x_ = torch.cat([x_masked, mask_tokens], dim=1)
    x_= torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,D)) # unshuffle

    num_masked = torch.sum(torch.sum(x_ - x.reshape(N, T*P, D), -1) == 0)
    assert num_masked == round(N*T*P*(1-mask_ratio))
    print('[passed]')

    #-------------------------------------------------------------

############ DRQ-v2-DMC ###############

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train_mode(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train_mode(state)
        return False

