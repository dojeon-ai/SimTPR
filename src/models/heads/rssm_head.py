import torch.nn as nn
import torch
from .base import BaseHead
from einops import rearrange
from torch.distributions import Normal
from src.common.train_utils import orthogonal_init, init_normalization, renormalize
from torch.distributions.kl import kl_divergence
from torch.nn.functional import mse_loss
import numpy as np


def fixup_init(layer, num_layers):
    nn.init.normal_(layer.weight, mean=0, std=np.sqrt(
        2 / (layer.weight.shape[0] * np.prod(layer.weight.shape[2:]))) * num_layers ** (-0.25))

class RSSMHead(BaseHead):
    name = 'rssm'
    def __init__(self, 
                 in_dim,
                 hid_dim,
                 deter,
                 state_dim,
                 min_std=0.1,
                 balance=0.8):
        super(RSSMHead, self).__init__()
        
        self._hidden = hid_dim
        self._deter = deter
        self._min_std=min_std
        self.state_dim = state_dim
        self._balance = balance
        
        # prior
        self.fc_state_embed = nn.Linear(state_dim, hid_dim)
        self.rssm = GRUCell(hid_dim, deter, norm=True)
        self.fc_rssm_hidden = nn.Linear(deter, hid_dim)
        self.fc_prior_dist = nn.Linear(hid_dim, 2 * state_dim)
        
        # posterior
        self.fc_rnn_hidden_embedded_obs = nn.Linear(deter + in_dim, hid_dim)
        self.fc_posterior_dist = nn.Linear(hid_dim, 2 * state_dim)
        
        # observation model
        self.obs_model =  ObsModel(in_dim, state_dim, deter, None)
        
        
    def prior(self, rnn_hidden, state):

        """
        h_t = f(h_t-1, z_t-1)
        Compute prior p(\hat{z}_t | h_t)
        """
        # use ELU (APV)
        # compute deterministic state h_t [Recurrent model in APV (Appendix B.1)]
        hidden = nn.ELU()(self.fc_state_embed(state))
        rnn_hidden = self.rssm(hidden, rnn_hidden)
        
        # compute prior using h_t => p(\hat{z}_t | h_t)
        # Transition predictor in APV (Appendix B.1)
        x = self.fc_rssm_hidden(rnn_hidden)
        x = nn.ELU()(x)
        mu, stddev = self.fc_prior_dist(x).chunk(2, dim = -1)
        stddev = nn.Softplus()(stddev) + self._min_std
        
        return Normal(mu, stddev), rnn_hidden
    
    def posterior(self, rnn_hidden, embedded_obs):
        """
        Compute posterior q(z_t | h_t, o_t)
        """
    
        # Use deterministic state h_t and image observations o_t to compute posterior stochastic states z_t
        # Representation model in APV (Appendix B.1)
        
        hidden = nn.ELU()(self.fc_rnn_hidden_embedded_obs(
            torch.cat([rnn_hidden, embedded_obs], dim=1)))
        
        mu, stddev = self.fc_posterior_dist(hidden).chunk(2, dim=-1)
        stddev = torch.nn.Softplus()(stddev) + self._min_std
        return Normal(mu, stddev)

    def forward(self, x):
        # \hat{z}_t ~ prior p(\hat{z}_t | h_t), h_t
        info = {}
        return x, info

    def kl_loss(self, x):
        n, t, d = x.shape
        device = x.device

        # x: (n,t,d) -> x: (t, n, d)
        x = x.transpose(0, 1)

        # prepare Tensor to maintain states sequence and rnn hidden states sequence
        states = torch.zeros(
                t, n, self.state_dim, device=device)
        rnn_hiddens = torch.zeros(
            t, n, self._deter, device=device)

        # initialize state and rnn hidden state with 0 vector
        state = torch.zeros(n, self.state_dim, device=device)
        rnn_hidden = torch.zeros(n, self._deter, device=device)

        # compute state and rnn hidden sequences and kl loss
        kl_loss = 0
        for l in range(t-1):
            next_state_prior, rnn_hidden = self.prior(rnn_hidden, state)
            next_state_posterior = self.posterior(rnn_hidden, x[l+1])

            state = next_state_posterior.rsample()
            states[l+1] = state
            rnn_hiddens[l+1] = rnn_hidden

            # APV => weighted sum of kld both sides

            kl_loss += kl_divergence(next_state_posterior, next_state_prior).sum(dim=1).clamp(min=0).mean()


        kl_loss /= (t - 1)


        return kl_loss, states, rnn_hiddens
    

    def recon_loss(self, obs, states, rnn_hiddens):
        n, t, f, c, h, w = obs.shape

        obs = rearrange(obs, 'n t f c h w -> t n (f c) h w')
        # states, rnn_hiddens should be reshaped in trainer 
        # recon from time step 1 
        recon_observations = rearrange(self.obs_model(states, rnn_hiddens), '(n t) c h w -> t n c h w', n=n, t=t)

        recon_loss = 0.5 * mse_loss(recon_observations[1:], obs[1:], reduction='none').mean()
        
        return recon_loss




# use Layernorm after dense layer (APV)
class GRUCell(nn.Module):
    def __init__(self, input_size, hid_size, norm=False, update_bias=-1, **kwargs):
        super(GRUCell, self).__init__()
        self._size = hid_size
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(input_size + self._size, 3 * self._size, True)
        if norm:
            self._norm = nn.LayerNorm(3 * self._size)
        
    
    def forward(self, inputs, state):
        # return previous layer output, prev hidden state
        parts = self._layer(torch.cat([inputs, state], dim = -1))
        if self._norm:
            parts =  self._norm(parts)
        reset, cand, update = torch.chunk(parts, 3, -1)
        reset = nn.Sigmoid()(reset)
        cand = nn.Tanh()(reset * cand)
        update = nn.Sigmoid()(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output
    


# Upsampling
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
                 hid_dim,
                 state_dim,
                 rnn_hidden_dim,
                 norm_type):
        super(ObsModel, self).__init__()
  
        channels = [64, 64, 32, 4]
        expansion_ratio=2
        blocks_per_group=3
        strides = [2, 2, 3]
        
        layers = []
        num_layers = len(channels) * 3

        self.fc = nn.Linear(state_dim + rnn_hidden_dim, hid_dim)

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

    def forward(self, state, hidden):
        '''
        input
        x: (N, T, in_dim -> 3136)

        output
        reconstructed image x: (N X T, 4, 84, 84)
        '''
        x = torch.cat([state, hidden], dim = -1)
        x = self.fc(x)
        x = x.reshape(x.shape[0], 64, 7, 7)
        x = self.layers(x)

        return x
