from .base import BaseHead
from src.common.train_utils import orthogonal_init, init_normalization, renormalize
from src.models.backbones.cnn.impala import ResidualBlock, TransposeResidualBlock

import torch.nn as nn
import torch
from torch.distributions import Normal
from einops import rearrange
import numpy as np


class VAEHead(BaseHead):
    name = 'vae'
    def __init__(self, 
                 in_dim,
                 state_dim):
        super().__init__()
        self.state_dim = state_dim

        # posterior
        # weight init scaling issue?
        self.fc_mu = nn.Linear(in_dim, state_dim)
        self.fc_var = nn.Linear(in_dim, state_dim)

        # reconstruction
        self.decoder_input = nn.Linear(state_dim, in_dim)
        self.decode_latent = ObsModel('bn')

    def encode(self, x):
        """
        [param] x: (N, T, D)
        [return] posterior: (2, N, T, D)
        """
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return [mu, log_var]

    def reparameterize(self, mu, logvar) :
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x T X D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x T X D]
        :return: (Tensor) [B x T X D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mu

    def decode(self, z):
        x = self.decoder_input(z)
        x = self.decode_latent(x)

        return x

    def encode_decode(self, input_imgs):
        info = {}
        mu, log_var = self.encode(input_imgs)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input_imgs, mu, log_var]

    def forward(self, x):
        info = {}
        
        return x, info


class ObsModel(nn.Module):
    def __init__(self,
                 norm_type):
        super(ObsModel, self).__init__()
        """
        Currently, Obs model only supports Impala-M
        """
        channels = [64, 64, 32, 4]
        expansion_ratio=2
        blocks_per_group=3
        strides = [2, 2, 3]
        
        layers = []
        num_layers = len(channels) * 3
        for i in range(len(strides)):
            layers.append(TransposeResidualBlock(in_channels=channels[i], 
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

        self.layers = nn.Sequential(*layers)        

    def forward(self, x):
        '''
        [input] x: (n, t, in_dim=3136)
        [output] x: (n * t, 4, 84, 84)
        '''
        n, t, d = x.shape
        x = x.reshape(n * t, 64, 7, 7)
        x = self.layers(x)

        return x

