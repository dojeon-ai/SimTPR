import torch.nn as nn
import torch.nn.functional as F
import torch, numpy as np
from einops import rearrange
from .base import BaseHead
from src.models.layers import *
from src.common.train_utils import xavier_uniform_init, init_normalization


class DTHead(BaseHead):
    name = 'dt'
    def __init__(self, 
                 obs_shape, 
                 action_size, 
                 t_step, 
                 in_dim, 
                 proj_dim, 
                 dec_num_layers):
        
        super().__init__()
        self.t_step = t_step
        self.in_dim = in_dim
        self.proj_dim = proj_dim
        
        self.rtg_in = nn.Linear(1, proj_dim)
        self.obs_in = nn.Linear(in_dim, proj_dim)
        self.act_in = nn.Embedding(action_size, proj_dim)
         
        self.dec_norm = nn.LayerNorm(proj_dim)
        self.decoder = TransRtgDet(obs_shape=obs_shape, 
                                action_size=action_size,
                                hid_dim=proj_dim,
                                num_layers=dec_num_layers)
        proj_in_dim = proj_dim        
        self.act_predictor = nn.Sequential(nn.Linear(proj_dim, proj_dim), 
                                           nn.ReLU(), 
                                           nn.Linear(proj_dim, action_size))

    def decode(self, x):        
        n, t, _ = x['obs'].shape
        
        rtg = self.rtg_in(x['rtg'].unsqueeze(-1))
        obs = self.obs_in(x['obs'])
        act = self.act_in(x['act'])

        # decoding
        T = 3 * t
        obs = self.dec_norm(obs)
        attn_mask = 1 - torch.ones((n, T, T), device=(obs.device)).tril_()
        x = self.decoder(rtg, obs, act, attn_mask)

        # prediction
        x = x[:, torch.arange(t)*3+1, :]
        x = self.act_predictor(x)
        
        return x

    def forward(self, x):
        info = {}
        return (x, info)
