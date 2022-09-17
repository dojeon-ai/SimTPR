import torch.nn as nn
import torch
from einops import rearrange
from .base import BaseHead


class BYOLHead(BaseHead):
    name = 'byol'
    def __init__(self, 
                 process_type,
                 in_features,
                 hid_features,
                 out_features):
        super().__init__()
        self.process_type = process_type
        self.projector = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hid_features),
            nn.BatchNorm1d(num_features=hid_features),
            nn.ReLU(),
            nn.Linear(in_features=hid_features, out_features=out_features)
        )
        self.predictor = nn.Sequential(
            nn.Linear(in_features=out_features, out_features=out_features),
            nn.BatchNorm1d(num_features=out_features),
            nn.ReLU(),
            nn.Linear(in_features=out_features, out_features=out_features)
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