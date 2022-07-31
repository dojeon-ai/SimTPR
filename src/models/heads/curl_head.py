import torch.nn as nn
import torch
from .base import BaseHead


class CURLHead(BaseHead):
    name = 'curl'
    def __init__(self, 
                 in_features,
                 hid_features,
                 out_features):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hid_features),
            nn.BatchNorm1d(num_features=hid_features),
            nn.ReLU(),
            nn.Linear(in_features=hid_features, out_features=out_features)
        )
        self.predictor = nn.Sequential(
            nn.Linear(in_features=out_features, out_features=out_features),
        )

    def project(self, x):
        x = self.projector(x)
        return x

    def predict(self, x):
        x = self.predictor(x)
        return x

    def forward(self, x):
        x = self.project(x)
        x = self.predict(x)
        return x