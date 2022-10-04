import torch.nn as nn
import torch.nn.functional as F
import torch, numpy as np
from einops import rearrange
from .base import BaseHead
from src.models.layers import *
from src.common.train_utils import init_normalization
from src.common.vit_utils import get_1d_sincos_pos_embed_from_grid


class MLRHead(BaseHead):
    name = 'mlr'
    def __init__(self, 
                 obs_shape, 
                 action_size, 
                 t_step, 
                 in_dim, 
                 proj_dim, 
                 dec_type, 
                 dec_num_layers):
        super().__init__()
        self.t_step = t_step
        self.in_dim = in_dim
        self.proj_dim = proj_dim
        self.dec_type = dec_type
        self.mask_token = nn.Parameter(torch.zeros(1, 1, proj_dim))

        if dec_type == 'trans_det':
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
        
        self.act_predictor = nn.Sequential(nn.Linear(2 * proj_dim, proj_dim), 
                                           nn.ReLU(), 
                                           nn.Linear(proj_dim, action_size))

    def decode(self, x, act):
        n, t, d = x.shape
        x_0 = x[:, 0:1, :]
        
        if self.dec_type == 'trans_det':
            x_a = self.decoder(x, act)
            # mlr: prediction from a masked token
            # rssm: prediction from a act token 
            x_pred = x_a[:, torch.arange(t) * 2, :]
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
