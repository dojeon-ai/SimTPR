import torch.nn as nn
import torch
from .base import BaseHead
from einops import rearrange
from torch.distributions import Normal
from src.common.train_utils import orthogonal_init


class RSSM(BaseHead):
    name = 'rssm'
    def __init__(self, 
                 in_dim,
                 hid_dim,
                 deter,
                 state_dim,
                 out_dim,
                 min_std=0.1):
        super(RSSM, self).__init__()
        
        self._hidden = hid_dim
        self._deter = deter
        self._min_std=min_std
        
        self.projector = nn.Linear(in_dim, hid_dim)
        
        # prior
        self.fc_state_embed = nn.Linear(state_dim, hid_dim)
        self.rssm = GRUCell(hid_dim, deter, norm=True)
        self.fc_rssm_hidden = nn.Linear(deter, hid_dim)
        self.fc_prior_dist = nn.Linear(hid_dim, 2 * state_dim)
        
        # posterior
        self.fc_rnn_hidden_embedded_obs = nn.Linear(deter + hid_dim, hid_dim)
        self.fc_posterior_dist = nn.Linear(hid_dim, 2 * state_dim)
        
        
        # observation model
        self.obs_model =  ObsModel()
        
    def project(self, x):
        x = self.projector(x)
        # use ELU (APV) -> ReLU activation
        x = nn.ReLU(x)
        return x
        
    def prior(self, rnn_hidden, state):
        """
        h_t = f(h_t-1, z_t-1)
        Compute prior p(\hat{z}_t | h_t)
        """
        # use ELU (APV) -> ReLU activation
        # compute deterministic state h_t [Recurrent model in APV (Appendix B.1)]
        hidden = nn.ReLU()(self.fc_state_embed(state))
        rnn_hidden = self.rssm(hidden, rnn_hidden)
        
        # compute prior using h_t => p(\hat{z}_t | h_t)
        # Transition predictor in APV (Appendix B.1)
        x = self.fc_rssm_hidden(rnn_hidden)
        x = nn.ReLU(x)
        mu, stddev = self.fc_prior_dist(x).chunck(2, dim = -1)
        stddev = nn.Softplus()(stddev) + self._min_std
        
        return Normal(mu, stddev), rnn_hidden
    
    def posterior(self, rnn_hidden, embedded_obs):
        """
        Compute posterior q(z_t | h_t, o_t)
        """
    
        # Use deterministic state h_t and image observations o_t to compute posterior stochastic states z_t
        # Representation model in APV (Appendix B.1)
        
        hidden = nn.ReLU()(self.fc_rnn_hidden_embedded_obs(
            torch.cat([rnn_hidden, embedded_obs], dim=1)))
        
        mu, stddev = self.fc_posterior_dist(hidden).chunck(2, dim=-1)
        stddev = torch.nn.Softplus()(stddev) + self._min_stddev
        return Normal(mu, stddev)

    def forward(self, state, rnn_hidden, embedded_next_obs):
        # \hat{z}_t ~ prior p(\hat{z}_t | h_t), h_t
        info = {}
        
        next_state_prior, rnn_hidden = self.prior(rnn_hidden, state)
        
        # z_t ~ posterior q(z_t | h_t, o_t)
        next_state_posterior = self.posterior(rnn_hidden, embedded_next_obs)
        
        x = {'next_state_prior': next_state_prior,
             'next_state_posterior': next_state_posterior,
             'rnn_hidden': rnn_hidden}
        
        return x, info




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
    
class TransposeImpalaBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expansion_ratio):
        super(TransposeImpalaBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, 
                                       out_channels=out_channels, 
                                       kernel_size=3, 
                                       stride=stride,
                                       padding=1,
                                       output_padding=stride-1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.res1 = ResidualBlock(out_channels, expansion_ratio)
        self.res2 = ResidualBlock(out_channels, expansion_ratio)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.res1(x)
        x = self.res2(x)
        
        return x
    
    
class ObsModel(nn.Module):
    def __init__(self,
                 hid_dim,
                 state_dim,
                 rnn_hidden_dim,
                 expansion_ratio,
                 init_type):
        super().__init__()

        self.fc = nn.Linear(state_dim + rnn_hidden_dim, hid_dim)
        self.layers = nn.Sequential(
            TransposeImpalaBlock(in_channels=64, out_channels=64, expansion_ratio=expansion_ratio, stride=2),
            TransposeImpalaBlock(in_channels=64, out_channels=32, expansion_ratio=expansion_ratio, stride=2),
            TransposeImpalaBlock(in_channels=32, out_channels=32, expansion_ratio=expansion_ratio, stride=3),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        )
        
        if init_type == 'orthogonal':
            self.apply(orthogonal_init)

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(x.shape[0], 64, 7, 7)
        x = self.layers(x)
        return x