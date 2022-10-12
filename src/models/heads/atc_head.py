import torch.nn as nn
import torch
from .base import BaseHead
from einops import rearrange


class ATCHead(BaseHead):
    name = 'atc'
    def __init__(self, 
                 in_dim,
                 hid_dim,
                 out_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )
        self.bilinear = nn.Linear(out_dim, out_dim)

    def project(self, x):
        n, t, d = x.shape
        x = rearrange(x, 'n t d-> (n t) d')
        x = self.projector(x)
        x = rearrange(x, '(n t) d-> n t d', t=t)
        return x
        
    def predict(self, x):
        n, t, d = x.shape
        x = rearrange(x, 'n t d-> (n t) d')
        # residual predictor h_phi in ATC
        # http://proceedings.mlr.press/v139/stooke21a/stooke21a.pdf
        x = self.predictor(x) + x
        x = self.bilinear(x)
        x = rearrange(x, '(n t) d-> n t d', t=t)
        return x

    def forward(self, x):
        x = self.project(x)
        x = self.predict(x)
        info = {}

        return x, info