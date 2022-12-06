import torch.nn as nn
import torch.nn.functional as F
import torch, numpy as np
from einops import rearrange
from .base import BaseHead
from src.models.layers import *
from src.common.vit_utils import get_1d_sincos_pos_embed_from_grid
from src.common.train_utils import xavier_uniform_init, init_normalization


class GPTHead(BaseHead):
    name = 'gpt'
    def __init__(self, 
                 obs_shape, 
                 action_size, 
                 t_step, 
                 in_dim, 
                 proj_dim, 
                 pred_dim,
                 num_layers,
                 dropout):
        
        super().__init__()
        self.t_step = t_step
        self.in_dim = in_dim
        self.proj_dim = proj_dim
        
        self.obs_in = nn.Sequential(nn.Linear(in_dim, proj_dim, bias=False), 
                                    nn.BatchNorm1d(proj_dim), 
                                    nn.ReLU(), 
                                    nn.Linear(proj_dim, proj_dim, bias=False),
                                    nn.BatchNorm1d(proj_dim, affine=False))
        self.act_in = nn.Embedding(action_size, proj_dim)
        self.rew_in = nn.Linear(1, proj_dim) 
        self.rtg_in = nn.Linear(1, proj_dim)
        
        self.act_token = nn.Parameter(torch.zeros(1, 1, proj_dim))
        self.rew_token = nn.Parameter(torch.zeros(1, 1, proj_dim))
        self.rtg_token = nn.Parameter(torch.zeros(1, 1, proj_dim))
        
        max_t_step = 200
        self.pos_embed = nn.Parameter((torch.randn(1, max_t_step, proj_dim)), requires_grad=False)
        pos_embed = get_1d_sincos_pos_embed_from_grid(proj_dim, np.arange(max_t_step))
        self.pos_embed.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        self.decoder = Transformer(dim=proj_dim, 
                                   depth=num_layers, 
                                   heads=proj_dim//64, 
                                   mlp_dim=proj_dim*4, 
                                   dropout=dropout)
                                    
        self.obs_pred = nn.Sequential(nn.Linear(proj_dim, pred_dim, bias=False), 
                                      nn.BatchNorm1d(pred_dim), 
                                      nn.ReLU(), 
                                      nn.Linear(pred_dim, proj_dim))
        self.act_pred = nn.Sequential(nn.Linear(proj_dim, proj_dim), 
                                      nn.ReLU(), 
                                      nn.Linear(proj_dim, action_size))
        # assume that reward lies under (-1~1)
        self.rew_pred = nn.Sequential(nn.Linear(proj_dim, proj_dim), 
                                      nn.ReLU(), 
                                      nn.Linear(proj_dim, 1),
                                      nn.Tanh()) 
        self.rtg_pred = nn.Sequential(nn.Linear(proj_dim, proj_dim), 
                                      nn.ReLU(), 
                                      nn.Linear(proj_dim, 1))
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.act_token, std=.02)
        torch.nn.init.normal_(self.rew_token, std=.02)
        torch.nn.init.normal_(self.rtg_token, std=.02)
        
        # initialize nn.Linear and nn.LayerNorm
        self.apply(xavier_uniform_init)
        
        
    def encode_obs(self, obs):
        n, t, d = obs.shape
        obs = rearrange(obs, 'n t d-> (n t) d')
        obs = self.obs_in(obs)
        obs = rearrange(obs, '(n t) d-> n t d', t=t)
        
        return obs
            
        
    def decode(self, obs, act, rew, rtg, act_mask, rew_mask, rtg_mask):
        n, t, d = obs.shape
        T = 4*t
        
        # embed
        obs = obs + self.pos_embed[:, :t, :]
        act = self.act_in(act) + self.pos_embed[:, :t, :]
        rew = self.rew_in(rew.unsqueeze(-1)) + self.pos_embed[:, :t, :]
        rtg = self.rtg_in(rtg.unsqueeze(-1)) + self.pos_embed[:, :t, :]
        
        # masking
        act_mask = act_mask.unsqueeze(-1)
        rew_mask = rew_mask.unsqueeze(-1)
        rtg_mask = rtg_mask.unsqueeze(-1)
        
        act = act_mask * self.act_token + (1-act_mask) * act
        rew = rew_mask * self.rew_token + (1-rew_mask) * rew
        rtg = rtg_mask * self.rtg_token + (1-rtg_mask) * rtg

        # construct input
        x = torch.zeros((n, T, d), device=(obs.device))
        x[:, torch.arange(t) * 4, :] += obs
        x[:, torch.arange(t) * 4 + 1, :] += act
        x[:, torch.arange(t) * 4 + 2, :] += rew
        x[:, torch.arange(t) * 4 + 3, :] += rtg
        
        # decode
        attn_mask = 1 - torch.ones((n, T, T), device=(x.device)).tril_()
        x, _ = self.decoder(x, attn_mask=attn_mask)
        
        # construct output
        obs = x[:, torch.arange(t)*4+3, :] # o_(t+1), ... o_(T+1)
        act = x[:, torch.arange(t)*4, :]   # a_(t), ... a_(T)
        rew = x[:, torch.arange(t)*4+1, :] # r_(t), ... r_(T)
        rtg = x[:, torch.arange(t)*4+2, :] # R_(t), ... R_(T)
    
        return obs, act, rew, rtg
    

    def predict(self, obs, act, rew, rtg):
        n, t, d = obs.shape
        obs = rearrange(obs, 'n t d-> (n t) d')
        obs = self.obs_pred(obs)
        obs = rearrange(obs, '(n t) d-> n t d', t=t)
        
        act = self.act_pred(act)
        rew = self.rew_pred(rew)
        rtg = self.rtg_pred(rtg)
        
        return obs, act, rew, rtg

    
    def forward(self, x):
        info = {}
        return (x, info)
