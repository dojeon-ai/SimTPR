import torch.nn as nn
import torch
from .base import BaseHeader


class SimCLRHeader(BaseHeader):
    name = 'simclr'
    def __init__(self, 
                 in_features,
                 hid_features,
                 out_features):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hid_features),
            nn.ReLU(),
            nn.Linear(in_features=hid_features, out_features=out_features)
        )

    def project(self, x):
        x = self.projector(x)
        return x

    def forward(self, x):
        x = self.project(x)
        return x