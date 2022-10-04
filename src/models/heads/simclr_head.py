import torch
import torch.nn as nn
from einops import rearrange
from .base import BaseHead


class SimCLRHead(BaseHead):
    name = 'simclr'
    def __init__(self, 
                 in_dim,
                 hid_dim,
                 out_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )

    def project(self, x):
        n, t, d = x.shape
        x = rearrange(x, 'n t d-> (n t) d')
        x = self.projector(x)
        x = rearrange(x, '(n t) d-> n t d', t=t)
        return x

    def forward(self, x):
        x = self.project(x)
        info = {}
        
        return x, info