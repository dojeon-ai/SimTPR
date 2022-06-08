import torch.nn as nn
import torch
from .base import BaseBackbone
from src.common.train_utils import orthogonal_init, xavier_uniform_init


class DENature(BaseBackbone):
    name = 'de_nature'
    def __init__(self,
                 obs_shape,
                 renormalize,
                 init_type):
        super().__init__()
        S, C, H, W = obs_shape
        in_channels = S * C

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=5), 
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=5), 
            nn.ReLU(),
            nn.Flatten()
        )
        self._output_dim = 576
        self.renormalize = renormalize

    def forward(self, x):
        x = self.layers(x)
        if self.renormalize:
            x = self._renormalize(x)
        return x
