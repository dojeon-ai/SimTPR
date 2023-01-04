import numpy as np
import torch.nn as nn
import torch
from einops import rearrange
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn.functional import mse_loss

from .base import BaseHead
from src.common.train_utils import orthogonal_init, init_normalization, renormalize
from src.models.backbones.cnn.impala import ResidualBlock, TransposeResidualBlock


class RSSMHead(BaseHead):
    name = 'rssm'
    def __init__(self, 
                 in_dim,
                 hid_dim,
                 deter,
                 state_dim,
                 min_std=0.1):
        super(RSSMHead, self).__init__()
        self._hidden = hid_dim
        self._deter = deter
        self._min_std=min_std
        self.state_dim = state_dim
        
        # prior
        self.fc_state_embed = nn.Linear(state_dim, hid_dim)
        self.rssm = nn.GRUCell(hid_dim, deter) 
        self.fc_rssm_hidden = nn.Linear(deter, hid_dim)
        self.fc_prior_dist = nn.Linear(hid_dim, 2 * state_dim)
        
        # posterior
        self.fc_rnn_hidden_embedded_obs = nn.Linear(deter + in_dim, hid_dim)
        self.fc_posterior_dist = nn.Linear(hid_dim, 2 * state_dim)
        
        # observation model
        self.obs_model =  ObsModel(in_dim, state_dim, deter, 'bn')
        
    def prior(self, rnn_hidden, state):
        """
        h_t = f(h_t-1, z_t-1)
        Compute prior p(\hat{z}_t | h_t)
        """
        # use ELU (APV)
        # compute deterministic state h_t [Recurrent model in APV (Appendix B.1)]
        #hidden = nn.ELU()(self.fc_state_embed(state))
        hidden = nn.ReLU()(self.fc_state_embed(state))
        rnn_hidden = self.rssm(hidden, rnn_hidden)
        
        # compute prior using h_t => p(\hat{z}_t | h_t)
        # Transition predictor in APV (Appendix B.1)
        x = self.fc_rssm_hidden(rnn_hidden)
        x = nn.ReLU()(x)
        mu, stddev = self.fc_prior_dist(x).chunk(2, dim = -1)
        stddev = nn.Softplus()(stddev) + self._min_std
        
        return Normal(mu, stddev), rnn_hidden

    def posterior(self, rnn_hidden, embedded_obs):
        """
        Compute posterior q(z_t | h_t, o_t)
        """
        # Use deterministic state h_t and image observations o_t to compute posterior stochastic states z_t
        # Representation model in APV (Appendix B.1)
        hidden = nn.ReLU()(self.fc_rnn_hidden_embedded_obs(
            torch.cat([rnn_hidden, embedded_obs], dim=-1)))

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
            kl_loss += kl_divergence(next_state_posterior, next_state_prior).sum(dim=-1).clamp(min=0).mean()

        kl_loss /= (t - 1)

        return kl_loss, states, rnn_hiddens

    def recon_loss(self, obs, states, rnn_hiddens):
        n, t, f, c, h, w = obs.shape

        obs = rearrange(obs, 'n t f c h w -> n t (f c) h w')
        recon = rearrange(self.obs_model(states, rnn_hiddens), '(n t) c h w -> n t c h w', n=n, t=t)
        recon_loss = mse_loss(recon[:, 1:], obs[:, 1:], reduction='none').mean([0, 1, 2]).sum()

        obs = obs[0, 1:, -1].unsqueeze(1)
        recon = recon.to(float)
        recon = torch.where(recon >= 1.0, 1.0, recon)
        recon = torch.where(recon < 0.0, 0.0, recon)[0, 1:, -1].unsqueeze(1)

        return recon_loss, recon, obs


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
