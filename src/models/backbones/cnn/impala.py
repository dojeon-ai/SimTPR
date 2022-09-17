import torch.nn as nn
import torch
import numpy as np
from einops import rearrange
from src.models.backbones.base import BaseBackbone
from src.common.train_utils import orthogonal_init, xavier_uniform_init, init_normalization


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_type):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            init_normalization(channels=in_channels, norm_type=norm_type),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            init_normalization(channels=in_channels, norm_type=norm_type)
        )
                
    def forward(self, x):
        out = self.layers(x)
        return out + x


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm_type):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=3, stride=stride, padding=1)
        self.res1 = ResidualBlock(out_channels, norm_type)
        self.res2 = ResidualBlock(out_channels, norm_type)

    def forward(self, x):
        x = self.conv(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class Impala(BaseBackbone):
    name = 'impala'
    def __init__(self,
                 obs_shape,
                 action_size,
                 process_type,
                 channels,
                 strides,
                 expansion_ratio,
                 norm_type,
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
  
        channels = np.array(eval(channels)) * expansion_ratio      
        channels = np.concatenate(([self.in_channel], channels))
        strides = eval(strides)
        
        layers = []
        for i in range(len(strides)):
            layers.append(ImpalaBlock(in_channels=channels[i], 
                                      out_channels=channels[i+1], 
                                      stride=strides[i],
                                      norm_type=norm_type))
        
        layers.append(nn.Flatten())
        self.layers = nn.Sequential(*layers)        
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

