import torch.nn as nn
import torch
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from src.common.train_utils import xavier_uniform_init
from src.common.vit_utils import get_1d_sincos_pos_embed_from_grid


#################################################
# Transformer
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
        if attn_mask is not None:
            dots.masked_fill_(attn_mask.unsqueeze(1).bool(), -1e9)
        
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'n h t d -> n t (h d)')
        out = self.to_out(out)
        return out, attn


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
        attn_maps = []
        for attn, ff in self.layers:
            attn_x, attn_map = attn(x, attn_mask=attn_mask)
            x = attn_x + x
            x = ff(x) + x
            attn_maps.append(attn_map)
            
        return x, attn_maps
    
    
#################################################
# Decoder
class GRUDet(nn.Module):
    def __init__(self, obs_shape, action_size, hid_dim, num_layers):
        super().__init__()
        act_dim = hid_dim // 4
        obs_act_dim = hid_dim + act_dim
        self.act_embedder = nn.Embedding(action_size, act_dim)
        self.norm_in = nn.LayerNorm(obs_act_dim)
        self.num_layers = num_layers
        self.decoder = nn.GRU(input_size=obs_act_dim, 
                              hidden_size=hid_dim,
                              num_layers=num_layers,
                              batch_first=True,
                              bidirectional=False)

    def forward(self, x, a, h):
        """
        [params] x: (n, t, d)
        [params] a: (n, t)
        [params] h: (n_l, n, d)
        """
        n, t, d = x.shape
        a = self.act_embedder(a)
        x = torch.cat([x, a], dim=(-1))
        x = self.norm_in(x)
        x, h = self.decoder(x, h)
        return (
         x, h)


class ConvDet(nn.Module):
    def __init__(self, obs_shape, action_size, hid_dim, num_layers, in_dim=3136):
        super().__init__()
        self.h, self.w = (7, 7)
        in_channel = in_dim // (self.h * self.w)
        self.action_size = action_size
        conv1 = nn.Conv2d(in_channel + action_size, hid_dim, 3, 1, 1)
        conv2 = nn.Conv2d(hid_dim, in_channel, 3, 1, 1)
        self.decoder = nn.Sequential(conv1, 
                                     init_normalization(channels=hid_dim, norm_type='bn'), 
                                     nn.ReLU(), 
                                     conv2)

    def forward(self, x, a):
        """
        [params] x: (n, t, d)
        [params] a: (n, t)
        """
        n, t, d = x.shape
        act_one_hot = F.one_hot((a.detach()), num_classes=(self.action_size))
        act_one_hot = rearrange(act_one_hot, 'n t a_d -> n t a_d 1 1')
        act_one_hot = act_one_hot.repeat((1, 1, 1, self.h, self.w))
        x = rearrange(x, 'n t (c h w) -> n t c h w', h=(self.h), w=(self.w))
        x = torch.cat([x, act_one_hot], dim=2)
        x = rearrange(x, 'n t c h w -> (n t) c h w')
        x = self.decoder(x)
        x = rearrange(x, '(n t) c h w -> n t (c h w)', t=t)
        return x


class TransDet(nn.Module):
    def __init__(self, obs_shape, action_size, hid_dim, num_layers):
        super().__init__()
        num_heads = hid_dim // 64
        mlp_dim = hid_dim * 4
        max_t_step = 256
        self.decoder = Transformer(dim=hid_dim, 
                                   depth=num_layers,
                                   heads=num_heads,
                                   mlp_dim=mlp_dim,
                                   dropout=0.0)
        self.norm_out = nn.LayerNorm(hid_dim)
        self.pos_embed = nn.Parameter((torch.randn(1, max_t_step, hid_dim)), requires_grad=False)
        pos_embed = get_1d_sincos_pos_embed_from_grid(hid_dim, np.arange(max_t_step))
        self.pos_embed.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.apply(xavier_uniform_init)

    def forward(self, obs, act=None, rew=None, rtg=None, 
                      attn_mask=None, dataset_type='demonstration'):
        """
        [params] obs: (n, t, d)
        [params] act: (n, t)
        [params] rew: (n, t)
        [returns] x: (n, T, d) 
           if act is not None: T=2*t
           if rew and rtg is not None: T=4*t
        """
        n, t, d = obs.shape

        if dataset_type == 'video':
            x = obs + self.pos_embed[:, :t, :]
            
        elif dataset_type == 'demonstration':
            if act is None:
                raise ValueError('requires action for demonstration data')
            
            obs = obs + self.pos_embed[:, :t, :]
            act = act + self.pos_embed[:, :t, :]
            x = torch.zeros((n, 2 * t, d), device=(obs.device))
            x[:, torch.arange(t) * 2, :] += obs
            x[:, torch.arange(t) * 2 + 1, :] += act
            
        elif dataset_type == 'trajectory':
            if act is None:
                raise ValueError('requires action for trajectory data')
                
            if rew is None:
                raise ValueError('requires reward for trajectory data')
                
            if rtg is None:
                raise ValueError('requires return-to-go for trajectory data')
            
            obs = obs + self.pos_embed[:, :t, :]
            act = act + self.pos_embed[:, :t, :]
            rew = rew + self.pos_embed[:, :t, :]
            rtg = rtg + self.pos_embed[:, :t, :]
            
            x = torch.zeros((n, 4 * t, d), device=(obs.device))
            x[:, torch.arange(t) * 4, :] += obs
            x[:, torch.arange(t) * 4 + 1, :] += act
            x[:, torch.arange(t) * 4 + 2, :] += rew
            x[:, torch.arange(t) * 4 + 3, :] += rtg
        
        x, _ = self.decoder(x, attn_mask=attn_mask)
        x = self.norm_out(x)
        
        return x
    
    
class TransRtgDet(nn.Module):
    def __init__(self, obs_shape, action_size, hid_dim, num_layers):
        super().__init__()
        num_heads = hid_dim // 64
        mlp_dim = hid_dim * 4
        max_t_step = 256
        self.decoder = Transformer(dim=hid_dim, 
                                   depth=num_layers,
                                   heads=num_heads,
                                   mlp_dim=mlp_dim,
                                   dropout=0.0)
        self.norm_out = nn.LayerNorm(hid_dim)
        self.pos_embed = nn.Parameter((torch.randn(1, max_t_step, hid_dim)), requires_grad=False)
        pos_embed = get_1d_sincos_pos_embed_from_grid(hid_dim, np.arange(max_t_step))
        self.pos_embed.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.apply(xavier_uniform_init)

    def forward(self, rtg, obs, act, attn_mask=None):
        """
        [params] rtg: (n, t)
        [params] obs: (n, t, d)
        [params] act: (n, t)
        [returns] x: (n, 3*t, d) 
        """
        n, t, d = obs.shape

        rtg = rtg + self.pos_embed[:, :t, :]
        obs = obs + self.pos_embed[:, :t, :]
        act = act + self.pos_embed[:, :t, :]
        
        x = torch.zeros((n, 3 * t, d), device=(obs.device))
        x[:, torch.arange(t) * 3, :] += rtg
        x[:, torch.arange(t) * 3 + 1, :] += obs
        x[:, torch.arange(t) * 3 + 2, :] += act
        
        x, _ = self.decoder(x, attn_mask=attn_mask)
        x = self.norm_out(x)
        
        return x
    