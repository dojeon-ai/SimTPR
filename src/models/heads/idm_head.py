import torch.nn as nn
import torch
from einops import rearrange
from .base import BaseHead


class IDMHead(BaseHead):
    name = 'idm'
    def __init__(self, 
                 in_dim,
                 hid_dim,
                 action_size):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(2*in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, action_size)
        )
        
    def predict(self, x):
        n, t, d = x.shape
        x = rearrange(x, 'n t d-> (n t) d')
        x = self.predictor(x)
        x = rearrange(x, '(n t) d-> n t d', t=t)
        return x
    
    def forward(self, x):
        info = {}

        return x, info