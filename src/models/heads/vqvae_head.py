import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import BaseHead
from src.models.backbones.impala import ResidualBlock

############################
# Codebook

class Codebook(nn.Module):
    def __init__(self, n_codes, embedding_dim):
        super().__init__()
        self.register_buffer('embeddings', torch.randn(n_codes, embedding_dim))
        self.register_buffer('N', torch.zeros(n_codes))
        self.register_buffer('z_avg', self.embeddings.data.clone())

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True
        
    def _tile(self, x):
        """
        [param] z: (N, D)
        """
        n, d = x.shape
        if n < self.n_codes:
            n_repeats = (self.n_codes + n - 1) // n
            std = 0.01 / np.sqrt(d)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        """
        [param] z: (N, D)
        """
        self._need_init = False
        
        # tile the representation if needed
        z = self._tile(z)
        
        # assign embeddings
        n = z.shape[0]
        _k_rand = z[torch.randperm(n)][:self.n_codes]
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))
        
    def forward(self, z):
        """
        [param] z: (N, D)
        """
        # embed in l2-normalized space
        z = F.normalize(z, p=2)
        
        # init embedding
        if self._need_init and self.training:
            self._init_embeddings(z)
            
        # assign code based on distance
        # cosine sim is identical to the distance in l2-normalized space (but faster computation!)
        distances = 1 - F.cosine_similarity(z.unsqueeze(1), self.embeddings.unsqueeze(0), dim=-1)
        encode_indices = torch.argmin(distances, 1)
        encode_onehot = F.one_hot(encode_indices, self.n_codes).type_as(z)
        embeddings = F.embedding(encode_indices, self.embeddings)
        
        # get commitment_loss
        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())
        
        # EMA codebook update
        if self.training:
            n_total = encode_onehot.sum(dim=0)
            encode_sum = z.T.matmul(encode_onehot)
            
            # N = 0.99 * N + 0.01 * n_total
            # z_avg = 0.99 * z_avg + 0.01 * encode_sum
            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.T, alpha=0.01)
            
            # update codebook
            n = self.N.sum()
            weights = ((self.N + 1e-7) / (n + self.n_codes * 1e-7)) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)
            
            # add random noise to the unused code vectors
            y = self._tile(z)
            _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
            usage = (self.N.view(self.n_codes, 1) >= 1).float()
            self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))
        
        # add z to receive gradient identical to the embeddings
        embeddings = (embeddings - z).detach() + z

        # measure perplexity
        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return {'embeddings': embeddings,
                'encodings': encode_indices,
                'commitment_loss': commitment_loss,
                'perplexity': perplexity}
    

class VQVAEHead(BaseHead):
    name = 'vq_vae'
    def __init__(self, 
                 obs_shape,
                 action_size,
                 n_codes,
                 in_features, 
                 embedding_dim,
                 dec_type,
                 dec_input_shape,
                 dec_init_type,
                 expansion_ratio):
        super().__init__()
        self.proj_in = nn.Linear(in_features, embedding_dim)
        self.codebook = Codebook(n_codes, embedding_dim)
        self.proj_out = nn.Linear(embedding_dim, in_features)
        
        if dec_type == 'impala':
            self.decoder = TransposeImpala(obs_shape,
                                           action_size,
                                           expansion_ratio,
                                           dec_init_type)

    def encode(self, x):
        """
        [param] x: (N, D)
        [return] embeddings
        [return] encodings
        """
        z = self.proj_in(x)
        vq_out = self.codebook(z)
        
        return vq_out['embeddings'], vq_out['encodings']
        
    def forward(self, x):
        """
        [param] x: (N, D)
        """
        # encode 
        x = self.proj_in(x)
        vq_out = self.codebook(x)
        x = self.proj_out(vq_out['embeddings'])
        
        
        
        
        
        
        import pdb
        pdb.set_trace()
        return x