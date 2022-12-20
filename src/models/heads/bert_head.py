import torch.nn as nn
import torch.nn.functional as F
import torch, numpy as np
from einops import rearrange
from .base import BaseHead
from src.models.layers import *
from src.common.vit_utils import get_1d_sincos_pos_embed_from_grid
from src.common.train_utils import init_normalization


class BERTHead(BaseHead):
    name = 'bert'
    def __init__(self, 
                 obs_shape, 
                 action_size, 
                 t_step, 
                 in_dim, 
                 proj_dim, 
                 pred_dim,
                 proj_bn,
                 pred_bn,
                 num_layers,
                 dropout):
        
        super().__init__()
        self.t_step = t_step
        self.in_dim = in_dim
        self.proj_dim = proj_dim
        
        self.obs_mask_token = nn.Parameter(torch.zeros(1, 1, proj_dim))
        self.act_mask_token = nn.Parameter(torch.zeros(1, 1, proj_dim))
        
        if proj_bn is True:
            self.obs_in = nn.Sequential(nn.Linear(in_dim, proj_dim, bias=False), 
                                        nn.BatchNorm1d(proj_dim), 
                                        nn.ReLU(), 
                                        nn.Linear(proj_dim, proj_dim))
        else:
            self.obs_in = nn.Sequential(nn.Linear(in_dim, proj_dim, bias=False), 
                                        nn.ReLU(), 
                                        nn.Linear(proj_dim, proj_dim))
        self.act_in = nn.Embedding(action_size, proj_dim)
        
        max_t_step = t_step*2
        self.pos_embed = nn.Parameter((torch.randn(1, max_t_step, proj_dim)), requires_grad=False)
        pos_embed = get_1d_sincos_pos_embed_from_grid(proj_dim, np.arange(max_t_step))
        self.pos_embed.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        self.decoder = Transformer(dim=proj_dim, 
                                   depth=num_layers, 
                                   heads=proj_dim//64, 
                                   mlp_dim=proj_dim*4, 
                                   dropout=dropout)
                                    
        if pred_bn is True:
            self.obs_pred = nn.Sequential(nn.Linear(proj_dim, pred_dim, bias=False), 
                                          nn.BatchNorm1d(pred_dim), 
                                          nn.ReLU(), 
                                          nn.Linear(pred_dim, proj_dim))
        else:
            self.obs_pred = nn.Sequential(nn.Linear(proj_dim, pred_dim, bias=False), 
                                          nn.ReLU(), 
                                          nn.Linear(pred_dim, proj_dim))
        self.act_pred = nn.Sequential(nn.Linear(proj_dim, proj_dim), 
                                      nn.ReLU(), 
                                      nn.Linear(proj_dim, action_size))
        self._initialize_weights()
        
        
    def _initialize_weights(self):
        torch.nn.init.normal_(self.obs_in[-1].weight, std=.02)
        torch.nn.init.normal_(self.obs_pred[-1].weight, std=.02)
        torch.nn.init.normal_(self.obs_mask_token, std=.02)
        torch.nn.init.normal_(self.act_mask_token, std=.02)
        
    
    def encode_obs(self, obs):
        n, t, d = obs.shape
        obs = rearrange(obs, 'n t d-> (n t) d')
        obs = self.obs_in(obs)
        obs = rearrange(obs, '(n t) d-> n t d', t=t)
        
        return obs
        

    def decode(self, obs, act, obs_mask, act_mask, dataset_type):        
        n, t, d = obs.shape        

        obs_mask = obs_mask.unsqueeze(-1)
        obs = obs * (1-obs_mask) + self.obs_mask_token * obs_mask
        obs = obs + self.pos_embed[:, :t, :]
        
        act = self.act_in(act)
        act_mask = act_mask.unsqueeze(-1)
        act = act * (1-act_mask) + self.act_mask_token * act_mask
        act = act + self.pos_embed[:, :t, :]
        
        if dataset_type == 'video':
            T = t
            x = obs
            
        elif dataset_type == 'demonstration':
            T = 2*t
            x = torch.zeros((n, T, d), device=(obs.device))
            x[:, torch.arange(t) * 2, :] += obs
            x[:, torch.arange(t) * 2 + 1, :] += act
        
        # decode
        attn_mask = 1 - torch.ones((n, T, T), device=(x.device)).tril_()
        x, _ = self.decoder(x, attn_mask=attn_mask)
        
        if dataset_type == 'video':
            obs = x
            act = torch.zeros_like(x)
            
        elif dataset_type == 'demonstration':
            obs = x[:, torch.arange(t) * 2, :] # o_(t+1), ... o_(T+1)
            act = x[:, torch.arange(t) * 2, :]   # a_(t), ... a_(T)
    
        return obs, act


    def predict(self, obs, act):
        n, t, d = obs.shape
        obs = rearrange(obs, 'n t d-> (n t) d')
        obs = self.obs_pred(obs)
        obs = rearrange(obs, '(n t) d-> n t d', t=t)
        
        act = self.act_pred(act)
        
        return obs, act

    
    def forward(self, x):
        info = {}
        return (x, info)
