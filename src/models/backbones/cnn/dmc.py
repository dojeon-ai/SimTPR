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
        t, c, h, w = obs_shape
        assert process_type in {'indiv_frame', 'stack_frame'}
        self.process_type = process_type
        if process_type == 'indiv_frame':
            self.in_channel = c
        elif process_type == 'stack_frame':
            self.in_channel = t * c
        
        self.laeyrs = nn.Sequential(
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
        n, t, _, _, _ = x.shape
        if self.process_type == 'indiv_frame':
            x = rearrange(x, 'n t c h w -> (n t) c h w')
        elif self.process_type == 'stack_frame':
            x = rearrange(x, 'n t c h w -> n (t c) h w')
        
        x = self.layers(x)
        
        if self.process_type == 'indiv_frame':
            x = rearrange(x, '(n t) d -> n t d', t=t)
        elif self.process_type == 'stack_frame':
            x = rearrange(x, 'n d -> n 1 d')
            
        info = {}
            
        return x, info
