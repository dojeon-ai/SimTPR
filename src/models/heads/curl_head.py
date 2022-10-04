import torch.nn as nn
import torch
from .base import BaseHead
from einops import rearrange


class CURLHead(BaseHead):
    name = 'curl'
    def __init__(self, 
                 in_dim,
                 hid_dim,
                 out_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(out_dim, out_dim)
        )

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

        return x, info