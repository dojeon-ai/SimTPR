import torch.nn as nn
import torch
from .base import BaseHeader


class MLPHeader(BaseHeader):
    name = 'mlp'
    def __init__(self, 
                 in_features,
                 hid_features,
                 out_features):
        super().__init__()
        self.header = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hid_features),
            nn.ReLU(),
            nn.Linear(in_features=hid_features, out_features=out_features)
        )

    def forward(self, x):
        x = self.header(x)
        return x