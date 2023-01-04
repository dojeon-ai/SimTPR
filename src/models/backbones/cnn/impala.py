import torch.nn as nn
import torch
import numpy as np
from einops import rearrange
from src.models.backbones.base import BaseBackbone
from src.common.train_utils import orthogonal_init, init_normalization, renormalize


def fixup_init(layer, num_layers):
    nn.init.normal_(layer.weight, mean=0, std=np.sqrt(
        2 / (layer.weight.shape[0] * np.prod(layer.weight.shape[2:]))) * num_layers ** (-0.25))
    

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 norm_type,
                 expansion_ratio,
                 num_layers):
        super(ResidualBlock, self).__init__()
        hid_channels = in_channels * expansion_ratio
        self.stride = stride
        if stride != 1:
            self.down = nn.Conv2d(in_channels, out_channels, stride, stride)
            fixup_init(self.down, 1)
        
        self.layers = []
        if expansion_ratio == 1:
            conv1 = nn.Conv2d(in_channels, hid_channels, 3, stride, 1, groups=hid_channels)
            conv2 = nn.Conv2d(hid_channels, in_channels, 1, 1, 0)
            fixup_init(conv1, num_layers)
            fixup_init(conv2, num_layers)
            
            self.layers = nn.Sequential(
                conv1, 
                init_normalization(channels=hid_channels, norm_type=norm_type),
                nn.ReLU(),
                conv2,
                init_normalization(channels=in_channels, norm_type=norm_type),
            )
        else:
            hid_channels = in_channels * expansion_ratio
            conv1 = nn.Conv2d(in_channels, hid_channels, 1, 1, 0)
            conv2 = nn.Conv2d(hid_channels, hid_channels, 3, stride, 1, groups=hid_channels)
            conv3 = nn.Conv2d(hid_channels, out_channels, 1, 1, 0)
            fixup_init(conv1, num_layers)
            fixup_init(conv2, num_layers)
            fixup_init(conv3, num_layers)
            
            self.layers = nn.Sequential(
                conv1,
                init_normalization(channels=hid_channels, norm_type=norm_type),
                nn.ReLU(),
                conv2, 
                init_normalization(channels=hid_channels, norm_type=norm_type),
                nn.ReLU(),
                conv3,
                init_normalization(channels=out_channels, norm_type=norm_type),
            )
        if norm_type is not None:
             nn.init.constant_(self.layers[-1].weight, 0)
                
    def forward(self, x):
        if self.stride != 1:
            identity = self.down(x)
        else:
            identity = x
        out = self.layers(x)

        return identity + out


class Impala(BaseBackbone):
    name = 'impala'
    def __init__(self,
                 obs_shape,
                 action_size,
                 channels,
                 strides,
                 scale_ratio,
                 expansion_ratio,
                 blocks_per_group,
                 norm_type,
                 init_type,
                 renormalize):
        super().__init__()
        self.obs_shape = obs_shape
        f, c, h, w = obs_shape
        self.in_channel = f * c
  
        channels = np.array(eval(channels)) * scale_ratio      
        channels = np.concatenate(([self.in_channel], channels))
        strides = eval(strides)
        
        layers = []
        num_layers = len(channels) * blocks_per_group
        for i in range(len(strides)):
            layers.append(ResidualBlock(in_channels=channels[i], 
                                        out_channels=channels[i+1], 
                                        expansion_ratio=expansion_ratio,
                                        stride=strides[i],
                                        norm_type=norm_type,
                                        num_layers=num_layers))
            
            for _ in range(1, blocks_per_group):
                layers.append(ResidualBlock(in_channels=channels[i+1], 
                                            out_channels=channels[i+1], 
                                            expansion_ratio=expansion_ratio,
                                            stride=1,
                                            norm_type=norm_type,
                                            num_layers=num_layers))        
        layers.append(nn.Flatten())        
        self.layers = nn.Sequential(*layers)        
        if init_type == 'orthogonal':
            self.apply(orthogonal_init)
        self.renormalize = renormalize

    def forward(self, x):
        n, t, f, c, h, w = x.shape
        x = rearrange(x, 'n t f c h w -> (n t) (f c) h w')
        x = self.layers(x)
        if self.renormalize:
            x = renormalize(x)
        x = rearrange(x, '(n t) d -> n t d', t=t)
        info = {}
            
        return x, info


class TransposeResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion_ratio,
                 stride,
                 norm_type,
                 num_layers):
        super(TransposeResidualBlock, self).__init__()
        hid_channels = in_channels * expansion_ratio
        self.stride = stride
        self.down = self.conv = nn.ConvTranspose2d(in_channels=in_channels, 
                                       out_channels=out_channels, 
                                       kernel_size=3, 
                                       stride=stride,
                                       padding=1,
                                       output_padding=stride-1)
        fixup_init(self.down, 1)
        
        self.layers = []
        if expansion_ratio == 1:
            conv1 = nn.ConvTranspose2d(in_channels=hid_channels, 
                                       out_channels=hid_channels, 
                                       kernel_size=3, 
                                       stride=stride,
                                       padding=1,
                                       output_padding=stride-1,
                                       groups=hid_channels)
            conv2 = nn.Conv2d(hid_channels, in_channels, 1, 1, 0)
            fixup_init(conv1, num_layers)
            fixup_init(conv2, num_layers)
            
            self.layers = nn.Sequential(
                conv1, 
                init_normalization(channels=hid_channels, norm_type=norm_type),
                nn.ReLU(),
                conv2,
                init_normalization(channels=in_channels, norm_type=norm_type),
            )
        else:
            hid_channels = in_channels * expansion_ratio
            conv1 = nn.Conv2d(in_channels, hid_channels, 1, 1, 0)
            conv2 = nn.ConvTranspose2d(in_channels=hid_channels, 
                                       out_channels=hid_channels, 
                                       kernel_size=3, 
                                       stride=stride,
                                       padding=1,
                                       output_padding=stride-1,
                                       groups=hid_channels)
            conv3 = nn.Conv2d(hid_channels, out_channels, 1, 1, 0)
            fixup_init(conv1, num_layers)
            fixup_init(conv2, num_layers)
            fixup_init(conv3, num_layers)
            
            self.layers = nn.Sequential(
                conv1,
                init_normalization(channels=hid_channels, norm_type=norm_type),
                nn.ReLU(),
                conv2, 
                init_normalization(channels=hid_channels, norm_type=norm_type),
                nn.ReLU(),
                conv3,
                init_normalization(channels=out_channels, norm_type=norm_type),
            )
        if norm_type is not None:
             nn.init.constant_(self.layers[-1].weight, 0)
                
    def forward(self, x):
        out = self.layers(x)
        return self.down(x) + out