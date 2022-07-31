import torch.nn as nn
import torch
from .base import BaseBackbone
from src.common.train_utils import orthogonal_init, xavier_uniform_init


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 expansion_ratio):
        super(ResidualBlock, self).__init__()
        if expansion_ratio == 1:
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU()
            )
        else:
            hid_channels = in_channels * expansion_ratio
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hid_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=hid_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=hid_channels, out_channels=hid_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=hid_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=hid_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU()
            )
                
    def forward(self, x):
        out = self.layers(x)
        return out + x


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.res1 = ResidualBlock(out_channels, expansion_ratio)
        self.res2 = ResidualBlock(out_channels, expansion_ratio)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class Impala(BaseBackbone):
    name = 'impala'
    def __init__(self,
                 obs_shape,
                 expansion_ratio,
                 renormalize,
                 init_type):
        super().__init__()
        S, C, H, W = obs_shape
        in_channels = S * C

        self.layers = nn.Sequential(
            ImpalaBlock(in_channels=in_channels, out_channels=32, expansion_ratio=expansion_ratio),
            ImpalaBlock(in_channels=32, out_channels=64, expansion_ratio=expansion_ratio),
            ImpalaBlock(in_channels=64, out_channels=64, expansion_ratio=expansion_ratio),
            torch.nn.AdaptiveMaxPool2d((7, 7)),
            nn.Flatten()
        )
        self._output_dim = 3136
        self.renormalize = renormalize
        if init_type == 'orthogonal':
            self.apply(orthogonal_init)
        self.bn = nn.BatchNorm2d(num_features=3136)

    def forward(self, x):
        x = self.layers(x)
        if self.renormalize:
            x = self._renormalize(x)
        return x
