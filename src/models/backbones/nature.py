import torch.nn as nn
import torch
from .base import BaseBackbone
from src.common.train_utils import orthogonal_init, xavier_uniform_init


class Nature(BaseBackbone):
    name = 'nature'
    def __init__(self,
                 obs_shape,
                 renormalize,
                 init_type):
        super().__init__()
        S, C, H, W = obs_shape
        in_channels = S * C

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), 
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), 
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), 
            nn.ReLU(),
            nn.Flatten()
        )
        self._output_dim = 3136
        self.renormalize = renormalize

    def forward(self, x):
        x = self.layers(x)
        if self.renormalize:
            x = self._renormalize(x)
        return x
