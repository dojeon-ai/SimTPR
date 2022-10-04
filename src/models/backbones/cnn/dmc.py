import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common.train_utils import weight_init
from src.models.backbones.base import BaseBackbone

class DMC(BaseBackbone):
    name='dmc'
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
            nn.Conv2d(in_channels, 32, 3, stride=2),
            nn.ReLU(), 
            nn.Conv2d(32, 32, 3, stride=1),                  
            nn.ReLU(), 
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(), 
            nn.Conv2d(32, 32, 3, stride=1),
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
