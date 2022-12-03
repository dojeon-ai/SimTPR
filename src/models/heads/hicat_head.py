import torch.nn as nn
import torch.nn.functional as F
import torch, numpy as np
from einops import rearrange
from .base import BaseHead
from src.models.layers import *
from src.common.vit_utils import get_1d_sincos_pos_embed_from_grid
from src.common.train_utils import xavier_uniform_init, init_normalization


class HiCaTHead(BaseHead):
    name = 'hicat'
    def __init__(self, 
                 obs_shape, 
                 action_size, 
                 t_step, 
                 in_dim, 
                 proj_dim, 
                 num_layers,
                 dropout):
        
        super().__init__()
        self.t_step = t_step
        self.in_dim = in_dim
        self.proj_dim = proj_dim
        
        self.obs_in = nn.Linear(in_dim, proj_dim)
        self.act_in = nn.Embedding(action_size, proj_dim)
        self.rew_in = nn.Linear(1, proj_dim) 
        self.rtg_in = nn.Linear(1, proj_dim)
        
        max_t_step = 200
        self.pos_embed = nn.Parameter((torch.randn(1, max_t_step, proj_dim)), requires_grad=False)
        pos_embed = get_1d_sincos_pos_embed_from_grid(proj_dim, np.arange(max_t_step))
        self.pos_embed.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        ##########################
        # state transformer
        self.s_decoder = Transformer(dim=proj_dim, 
                                     depth=num_layers, 
                                     heads=proj_dim//64, 
                                     mlp_dim=proj_dim*4, 
                                     dropout=dropout)
                                    
        self.s_obs_proj = nn.Sequential(nn.Linear(proj_dim, proj_dim), 
                                        nn.BatchNorm1d(proj_dim), 
                                        nn.ReLU(), 
                                        nn.Linear(proj_dim, proj_dim))
        self.s_obs_pred = nn.Sequential(nn.Linear(proj_dim, proj_dim), 
                                        nn.BatchNorm1d(proj_dim), 
                                        nn.ReLU(), 
                                        nn.Linear(proj_dim, proj_dim))
        
        ############################
        # demonstration transformer
        self.d_decoder = Transformer(dim=proj_dim, 
                                     depth=num_layers, 
                                     heads=proj_dim//64, 
                                     mlp_dim=proj_dim*4, 
                                     dropout=dropout)
        
        self.d_obs_proj = nn.Sequential(nn.Linear(proj_dim, proj_dim), 
                                        nn.BatchNorm1d(proj_dim), 
                                        nn.ReLU(), 
                                        nn.Linear(proj_dim, proj_dim))
        self.d_obs_pred = nn.Sequential(nn.Linear(proj_dim, proj_dim), 
                                        nn.BatchNorm1d(proj_dim), 
                                        nn.ReLU(), 
                                        nn.Linear(proj_dim, proj_dim))
        self.d_act_pred = nn.Sequential(nn.Linear(proj_dim, proj_dim), 
                                        nn.ReLU(), 
                                        nn.Linear(proj_dim, action_size))
        
        ###############################
        # trajectory transformer
        self.t_decoder = Transformer(dim=proj_dim, 
                                     depth=num_layers, 
                                     heads=proj_dim//64, 
                                     mlp_dim=proj_dim*4, 
                                     dropout=dropout)
        
        self.t_obs_pred = nn.Sequential(nn.Linear(proj_dim, proj_dim), 
                                        nn.ReLU(), 
                                        nn.Linear(proj_dim, proj_dim))
        self.t_act_pred = nn.Sequential(nn.Linear(proj_dim, proj_dim), 
                                        nn.ReLU(), 
                                        nn.Linear(proj_dim, action_size))
        # assume that reward lies under (-1~1)
        self.t_rew_pred = nn.Sequential(nn.Linear(proj_dim, proj_dim), 
                                        nn.ReLU(), 
                                        nn.Linear(proj_dim, 1),
                                        nn.Tanh()) 
        self.t_rtg_pred = nn.Sequential(nn.Linear(proj_dim, proj_dim), 
                                        nn.ReLU(), 
                                        nn.Linear(proj_dim, 1))
        
    def encode_obs(self, obs):
        n, t, d = obs.shape
        obs = self.obs_in(obs)
        
        return obs
    
    
    def decode_state(self, obs):
        n, t, d = obs.shape
        T = t
        
        x = obs + self.pos_embed[:, :t, :]
        #x = self.s_norm_in(x)
        
        attn_mask = 1 - torch.ones((n, T, T), device=(x.device)).tril_()
        x, _ = self.s_decoder(x, attn_mask=attn_mask)
        #x = self.s_norm_out(x)
        
        # prediction
        obs = x
        
        return obs
    
    
    def predict_state(self, obs):
        n, t, d = obs.shape
        obs = rearrange(obs, 'n t d-> (n t) d')
        obs = self.s_obs_proj(obs)
        obs = self.s_obs_pred(obs)
        obs = rearrange(obs, '(n t) d-> n t d', t=t)
        
        return obs
    
    
    def project_state(self, obs):
        n, t, d = obs.shape
        obs = rearrange(obs, 'n t d-> (n t) d')
        obs = self.s_obs_proj(obs)
        obs = rearrange(obs, '(n t) d-> n t d', t=t)
        
        return obs
    
    
    def decode_demonstration(self, obs, act):
        n, t, d = obs.shape
        T = 2*t
        
        obs = obs + self.pos_embed[:, :t, :]
        act = self.act_in(act) + self.pos_embed[:, :t, :]
        
        x = torch.zeros((n, T, d), device=(obs.device))
        x[:, torch.arange(t) * 2, :] += obs
        x[:, torch.arange(t) * 2 + 1, :] += act
        
        #x = self.d_norm_in(x)
        attn_mask = 1 - torch.ones((n, T, T), device=(x.device)).tril_()
        x, _ = self.d_decoder(x, attn_mask=attn_mask)
        #x = self.d_norm_out(x)
        
        obs = x[:, torch.arange(t)*2+1, :] # o_(t+1), ... o_(T+1)
        act = x[:, torch.arange(t)*2, :]   # a_(t), ... a_(T)
        
        return obs, act
        
    
    def predict_demonstration(self, obs, act):
        n, t, d = obs.shape
        obs = rearrange(obs, 'n t d-> (n t) d')
        obs = self.d_obs_proj(obs)
        obs = self.d_obs_pred(obs)
        obs = rearrange(obs, '(n t) d-> n t d', t=t)
        
        act = self.d_act_pred(act)
    
        return obs, act
    
    
    def project_demonstration(self, obs):
        n, t, d = obs.shape
        obs = rearrange(obs, 'n t d-> (n t) d')
        obs = self.d_obs_proj(obs)
        obs = rearrange(obs, '(n t) d-> n t d', t=t)
        
        return obs
    
    
    def decode_trajectory(self, obs, act, rew, rtg):
        n, t, d = obs.shape
        T = 4*t
        
        obs = obs + self.pos_embed[:, :t, :]
        act = self.act_in(act) + self.pos_embed[:, :t, :]
        rew = self.rew_in(rew.unsqueeze(-1)) + self.pos_embed[:, :t, :]
        rtg = self.rtg_in(rtg.unsqueeze(-1)) + self.pos_embed[:, :t, :]
        
        x = torch.zeros((n, T, d), device=(obs.device))
        x[:, torch.arange(t) * 4, :] += obs
        x[:, torch.arange(t) * 4 + 1, :] += act
        x[:, torch.arange(t) * 4 + 2, :] += rew
        x[:, torch.arange(t) * 4 + 3, :] += rtg
        
        #x = self.t_norm_in(x)
        attn_mask = 1 - torch.ones((n, T, T), device=(x.device)).tril_()
        x, _ = self.t_decoder(x, attn_mask=attn_mask)
        #x = self.t_norm_out(x)
        
        obs = x[:, torch.arange(t)*4+3, :] # o_(t+1), ... o_(T+1)
        act = x[:, torch.arange(t)*4, :]   # a_(t), ... a_(T)
        rew = x[:, torch.arange(t)*4+1, :] # r_(t), ... r_(T)
        rtg = x[:, torch.arange(t)*4+2, :] # R_(t), ... R_(T)
    
        return obs, act, rew, rtg
    
    
    def predict_trajectory(self, obs, act, rew, rtg):
        # no obs projection in trajectory
        n, t, d = obs.shape
        obs = rearrange(obs, 'n t d-> (n t) d')
        obs = self.t_obs_pred(obs)
        obs = rearrange(obs, '(n t) d-> n t d', t=t)
        
        act = self.t_act_pred(act)
        rew = self.t_rew_pred(rew)
        rtg = self.t_rtg_pred(rtg)
        
        return obs, act, rew, rtg
    
    
    def decode(self, obs, act, rew, rtg):
        n, t, d = obs.shape
        
        z = self.encode_obs(obs)
        s_obs = self.decode_state(z)
        
        d_obs = torch.cat((z[:, 0:1], s_obs[:, :-1]), 1)
        d_obs, d_act = self.decode_demonstration(d_obs, act)
        
        t_obs = torch.cat((z[:, 0:1], d_obs[:, :-1]), 1)
        t_obs, t_act, t_rew, t_rtg = self.decode_trajectory(t_obs, d_act, rew, rtg)
        
        obs, act, rew, rtg = self.predict_trajectory(t_obs, t_act, t_rew, t_rtg)
        
        return obs, act, rew, rtg
    

    def forward(self, x):
        info = {}
        return (x, info)
