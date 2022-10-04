import torch.nn as nn
import torch
from src.models.backbones.base import BaseBackbone
from src.common.train_utils import orthogonal_init, xavier_uniform_init


class DENature(BaseBackbone):
    name = 'de_nature'
    def __init__(self,
                 obs_shape,
                 action_size,
                 process_type,
                 init_type):
        super().__init__()
        self.obs_shape = obs_shape
        f, c, h, w = obs_shape
        self.in_channel = f * c

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=5), 
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=5), 
            nn.ReLU(),
            nn.Flatten()
        )
        if init_type == 'orthogonal':
            self.apply(orthogonal_init)

    def forward(self, x):
        n, t, f, c, h, w = x.shape
        x = rearrange(x, 'n t f c h w -> (n t) (f c) h w')
        x = self.layers(x)
        x = rearrange(x, '(n t) d -> n t d', t=t)
        info = {}
            
        return x, info