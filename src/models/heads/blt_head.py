import torch.nn as nn
import torch.nn.functional as F
import torch, numpy as np
from einops import rearrange
from .base import BaseHead
from src.models.layers import *
from src.common.train_utils import xavier_uniform_init, init_normalization


class BLTHead(BaseHead):
    name = 'blt'
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
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, proj_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        self.obs_in = nn.Linear(in_dim, proj_dim)
        self.act_in = nn.Embedding(action_size, proj_dim)
        self.rew_in = nn.Linear(1, proj_dim) 
        
        self.dec_norm = nn.LayerNorm(proj_dim)
        self.decoder = TransDet(obs_shape=obs_shape, 
                                action_size=action_size,
                                hid_dim=proj_dim,
                                num_layers=dec_num_layers)
        proj_in_dim = proj_dim
                    
        self.projector = nn.Sequential(nn.Linear(proj_in_dim, proj_dim), 
                                       nn.BatchNorm1d(proj_dim), 
                                       nn.ReLU(), 
                                       nn.Linear(proj_dim, proj_dim))
        
        self.predictor = nn.Sequential(nn.Linear(proj_dim, proj_dim), 
                                       nn.BatchNorm1d(proj_dim), 
                                       nn.ReLU(), 
                                       nn.Linear(proj_dim, proj_dim))
        
        self.act_predictor = nn.Sequential(nn.Linear(proj_dim, proj_dim), 
                                           nn.ReLU(), 
                                           nn.Linear(proj_dim, action_size))
        
        # assume that reward lies under (-1~1)
        self.rew_predictor = nn.Sequential(nn.Linear(proj_dim, proj_dim), 
                                           nn.ReLU(), 
                                           nn.Linear(proj_dim, 1),
                                           nn.Tanh()) 

    def decode(self, x, mask):        
        n, t, _ = x['obs'].shape
        
        # embedding
        # x = (o_1, a_1, r_1, o_2, a_2, r_2, ...)
        T = 3 * t
        obs = self.obs_in(x['obs'])
        act = self.act_in(x['act'])
        rew = self.rew_in(x['rew'].unsqueeze(-1))
        
        # mask
        obs_mask = mask['obs'].unsqueeze(-1)
        act_mask = mask['act'].unsqueeze(-1)
        rew_mask = mask['rew'].unsqueeze(-1)
        obs = obs * (1-obs_mask) + self.mask_token * obs_mask
        act = act * (1-act_mask) + self.mask_token * act_mask
        rew = rew * (1-rew_mask) + self.mask_token * rew_mask
        
        # forward
        obs = self.dec_norm(obs)
        act = self.dec_norm(act)
        rew = self.dec_norm(rew)
        x = self.decoder(obs, act, rew, attn_mask=None, dataset_type='trajectory')

        # prediction
        obs = x[:, torch.arange(t)*3, :] # o_(t+1), ... o_(T+1)
        act = x[:, torch.arange(t)*3+1, :]   # a_(t), ... a_(T)
        rew = x[:, torch.arange(t)*3+2, :] # r_(t), ... r_(T)
            
        act = self.act_predictor(act)
        rew = self.rew_predictor(rew)
        
        x = {'obs': obs,
             'act': act,
             'rew': rew}
        
        return x

    def project(self, x):
        n, t, d = x.shape
        x = rearrange(x, 'n t d-> (n t) d')
        if d != self.proj_dim:
            x = self.obs_in(x)
            x = self.dec_norm(x)
            
        x = self.projector(x)
        x = rearrange(x, '(n t) d-> n t d', t=t)
        return x

    def predict(self, x):
        n, t, d = x.shape
        x = rearrange(x, 'n t d-> (n t) d')
        x = self.predictor(x)
        x = rearrange(x, '(n t) d-> n t d', t=t)
        return x

    def forward(self, x):
        info = {}
        return (x, info)
