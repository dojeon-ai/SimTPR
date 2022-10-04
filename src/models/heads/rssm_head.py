import torch.nn as nn
import torch.nn.functional as F
import torch, numpy as np
from einops import rearrange
from .base import BaseHead
from src.models.layers import *
from src.common.train_utils import xavier_uniform_init, init_normalization


class RSSMHead(BaseHead):
    name = 'rssm'
    def __init__(self, 
                 obs_shape, 
                 action_size, 
                 t_step, 
                 in_dim, 
                 proj_dim, 
                 dec_strategy, 
                 dec_type, 
                 dec_num_layers):
        
        super().__init__()
        self.t_step = t_step
        self.in_dim = in_dim
        self.proj_dim = proj_dim
        self.dec_strategy = dec_strategy
        self.dec_type = dec_type
        if dec_type == 'gru_det':
            self.dec_in = nn.Linear(in_dim, proj_dim)
            self.decoder = GRUDet(obs_shape=obs_shape, 
                                  action_size=action_size,
                                  hid_dim=proj_dim,
                                  num_layers=dec_num_layers)
            proj_in_dim = proj_dim
            
        elif dec_type == 'conv_det':
            self.dec_in = nn.Identity()
            self.decoder = ConvDet(obs_shape=obs_shape, 
                                   action_size=action_size,
                                   hid_dim=proj_dim,
                                   in_dim=in_dim)
            proj_in_dim = in_dim
            
        elif dec_type == 'trans_det':
            self.dec_in = nn.Linear(in_dim, proj_dim)
            self.decoder = TransDet(obs_shape=obs_shape, 
                                    action_size=action_size,
                                    hid_dim=proj_dim,
                                    num_layers=dec_num_layers)
            proj_in_dim = proj_dim
            
        else:
            raise NotImplemented
                    
        self.projector = nn.Sequential(nn.Linear(proj_in_dim, proj_dim), 
                                       nn.BatchNorm1d(proj_dim), 
                                       nn.ReLU(), 
                                       nn.Linear(proj_dim, proj_dim), 
                                       nn.BatchNorm1d(proj_dim, affine=False))
        
        self.predictor = nn.Sequential(nn.Linear(proj_dim, proj_dim), 
                                       nn.BatchNorm1d(proj_dim), 
                                       nn.ReLU(), 
                                       nn.Linear(proj_dim, proj_dim))
        
        self.idm_predictor = nn.Sequential(nn.Linear(2 * proj_dim, proj_dim), 
                                           nn.ReLU(), 
                                           nn.Linear(proj_dim, action_size))
        
        self.bc_predictor = nn.Sequential(nn.Linear(2 * proj_dim, proj_dim), 
                                           nn.ReLU(), 
                                           nn.Linear(proj_dim, action_size))
        
        

    def decode(self, x, act):
        x = self.dec_in(x)
        n, t, d = x.shape
        x_0 = x[:, 0:1, :]
        if self.dec_strategy == 'ar':
            x_pred = []
            if self.dec_type == 'gru_det':
                n_l = self.decoder.num_layers
                x_t = x[:, 0:1, :]
                h_t = torch.zeros((n_l, n, d), device=(x.device))
                for t_idx in range(t):
                    a_t = act[:, t_idx:t_idx + 1]
                    x_t, h_t = self.decoder(x_t, a_t, h_t)
                    x_pred.append(x_t)

            elif self.dec_type == 'conv_det':
                x_t = x[:, 0:1, :]
                for t_idx in range(t):
                    a_t = act[:, t_idx:t_idx + 1]
                    x_t = self.decoder(x_t, a_t)
                    x_pred.append(x_t)

            elif self.dec_type == 'trans_det':
                x_0_to_t = x[:, 0:1, :]
                for t_idx in range(t):
                    a_0_to_t = act[:, :t_idx + 1]
                    x_a = self.decoder(x_0_to_t, a_0_to_t)
                    x_t = x_a[:, -1:]
                    x_0_to_t = torch.cat((x_0_to_t, x_t), 1)
                    x_pred.append(x_t)

            else:
                raise NotImplemented
            x_pred = torch.cat(x_pred, dim=1)
            
        if self.dec_strategy == 'tf':
            if self.dec_type == 'trans_det':
                attn_mask = 1 - torch.ones((n, 2 * t, 2 * t), device=(x.device)).tril_()
                x_a = self.decoder(x, act, attn_mask)
                x_pred = x_a[:, torch.arange(t) * 2 + 1, :]
            else:
                raise NotImplemented
                
        x = torch.cat((x_0, x_pred[:, :-1]), dim=1)
        return x

    def act_predict(self, x1, x2):
        x = torch.cat((x1, x2), -1)
        x = self.act_predictor(x)
        return x

    def project(self, x):
        n, t, d = x.shape
        x = rearrange(x, 'n t d-> (n t) d')
        if d != self.proj_dim:
            x = self.dec_in(x)
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
        x = self.project(x)
        x = self.predict(x)
        info = {}
        return (x, info)
