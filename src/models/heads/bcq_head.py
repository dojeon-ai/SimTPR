import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange
from .base import BaseHead


class BCQHead(BaseHead):
    name = 'bcq'
    def __init__(self, 
                 in_dim,
                 hid_dim,
                 action_size,
                 tau):
        super().__init__()
        self.tau = tau
        self.act_predictor = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, action_size)
        )
        self.q_predictor = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, action_size)
        )
        
    def act_predict(self, x):
        n, t, d = x.shape
        x = rearrange(x, 'n t d-> (n t) d')
        x = self.act_predictor(x)
        x = rearrange(x, '(n t) d-> n t d', t=t)
        return x
    
    def q_predict(self, x):
        n, t, d = x.shape
        x = rearrange(x, 'n t d-> (n t) d')
        x = self.q_predictor(x)
        x = rearrange(x, '(n t) d-> n t d', t=t)
        return x
    
    def forward(self, x):
        act_pred = self.act_predict(x)
        act_pred = rearrange(act_pred, 'n t d -> (n t) d')
        
        q_pred = self.q_predict(x)
        q_pred = rearrange(q_pred, 'n t d -> (n t) d')
        
        inf = 1e9
        act_prob = F.softmax(act_pred, -1)
        act_prob = act_prob / torch.max(act_prob, -1)[0].unsqueeze(-1)
        act_mask = (act_prob < self.tau) * inf
        q_pred = q_pred - act_mask
                
        info = {}

        return q_pred, info