import torch.nn as nn
import torch
from .base import BaseHead
from einops import rearrange
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn.functional import mse_loss
from src.common.train_utils import orthogonal_init, init_normalization, renormalize
import numpy as np

def fixup_init(layer, num_layers):
    nn.init.normal_(layer.weight, mean=0, std=np.sqrt(
        2 / (layer.weight.shape[0] * np.prod(layer.weight.shape[2:]))) * num_layers ** (-0.25))


class VAEHead(BaseHead):
    name = 'vae'
    def __init__(self, 
                 in_dim,
                 hid_dim):
        super().__init__()
        self.hid_dim = hid_dim

        # posterior
        # weight init scaling issue?
        self.fc_mu = nn.Linear(in_dim, hid_dim)
        self.fc_var = nn.Linear(in_dim, hid_dim)

        # reconstruction
        self.decoder_input = nn.Linear(hid_dim, in_dim)
        self.decode_latent = ObsModel('bn')

        
    def encode(self, x):
        """
        [param] x: (N, T, D)
        [return] posterior: (2, N, T, D)
        """
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return [mu, log_var]


    def forward(self, x):
        return x

    def encode_decode(self, input_imgs):
        info = {}
        mu, log_var = self.encode(input_imgs)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input_imgs, mu, log_var]

    def decode(self, z):
        x = self.decoder_input(z)
        x = self.decode_latent(x)

        return x

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

    def loss_function(self,
                      recons,
                      input_image,
                      mu,
                      log_var) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = recons
        input_image = input_image
        mu = mu
        log_var = log_var

        recons_loss = mse_loss(recons, input_image)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = -1))

        loss = recons_loss +  kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.hid_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples
    

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion_ratio,
                 stride,
                 norm_type,
                 num_layers):
        super(ResidualBlock, self).__init__()
        hid_channels = in_channels * expansion_ratio
        
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
        out = self.layers(x)
        return x + out


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


class ObsModel(nn.Module):
    def __init__(self,
                 norm_type):
        super(ObsModel, self).__init__()
  
        channels = [64, 64, 32, 4]
        expansion_ratio=2
        blocks_per_group=3
        strides = [2, 2, 3]
        
        layers = []
        num_layers = len(channels) * 3
        for i in range(len(strides)):
            for _ in range(1, blocks_per_group):
                layers.append(ResidualBlock(in_channels=channels[i], 
                                            out_channels=channels[i], 
                                            expansion_ratio=expansion_ratio,
                                            stride=1,
                                            norm_type=norm_type,
                                            num_layers=num_layers))

            layers.append(TransposeResidualBlock(in_channels=channels[i], 
                                        out_channels=channels[i+1], 
                                        expansion_ratio=expansion_ratio,
                                        stride=strides[i],
                                        norm_type=norm_type,
                                        num_layers=num_layers))      
     
        self.layers = nn.Sequential(*layers)        

    def forward(self, x):
        '''
        input
        x: (N, T, in_dim -> 3136)

        output
        reconstructed image x: (N X T, 4, 84, 84)
        '''
        n, t, d = x.shape
        x = x.reshape(n * t, 64, 7, 7)
        x = self.layers(x)

        return x

