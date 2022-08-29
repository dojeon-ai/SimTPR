import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common.train_utils import weight_init

import ipdb

from src.models.backbones.base import BaseBackbone

class DrQv2Encoder(BaseBackbone):
    name='drqv2_encoder'
    def __init__(self, 
                 obs_shape,
                 conv_filters,
                 kernel_sizes,
                 strides,
                 action_size
                 ):
        super().__init__()

        assert len(obs_shape) == 4 
        assert (len(conv_filters) == len(kernel_sizes))
        assert (len(conv_filters)  == len(strides))
        S, C, H, W = obs_shape  
        in_channels = S * C 
        self.convnet = nn.Sequential(nn.Conv2d(in_channels, 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        # calculate the dimension of the conv-output
        test_obs = torch.rand(obs_shape)
        test_obs = test_obs.reshape(1, S*C, H, W)
        output_shape = self.forward(test_obs).squeeze().shape
        assert len(output_shape) == 1, 'conv shape 오류'
        self.repr_features = output_shape[0]

        self.apply(weight_init)

    def forward(self, obs):
        obs = self.convnet(obs)
        h = obs.view(obs.shape[0], -1)

        return h
