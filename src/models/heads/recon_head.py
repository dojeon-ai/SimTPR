import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import BaseHead
from src.models.backbones.impala import TransposeImpala
from einops.layers.torch import Rearrange
from einops import rearrange
    

class ReconHead(BaseHead):
    name = 'recon'
    def __init__(self, 
                 obs_shape,
                 action_size,
                 enc_type,
                 hog_in_features,
                 hog_hid_features,
                 hog_out_features,
                 dec_input_shape,
                 dec_init_type,
                 expansion_ratio):
        super().__init__()
        if enc_type == 'impala':
            self.to_spatial = Rearrange('n (c h w) -> n c h w', h=7, w=7)
            self.patch_recon = nn.Sequential(
                TransposeImpala(obs_shape, action_size, expansion_ratio, dec_init_type),
                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
            )
            self.hog_pred = nn.Sequential(
                nn.Linear(in_features=hog_in_features, out_features=hog_hid_features),
                nn.ReLU(),
                nn.Linear(in_features=hog_hid_features, out_features=hog_out_features)
            )
            self.flow_pred = nn.Sequential(
                TransposeImpala(obs_shape, action_size, expansion_ratio, dec_init_type),
                nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1),
            )
        
    def forward(self, x):
        """
        [param] x: (N, D)
        [return] patch_pred: (N, C, H, W)
        [return] hog_pred: (N, H_D)
        [return] flow_pred: (N, 2, H, W)
        """
        n, t, d = x.shape
        
        # patch recon
        _x = rearrange(x, 'n t d -> (n t) d')
        spatial_x = self.to_spatial(_x)
        patch_pred = self.patch_recon(spatial_x)
        
        # hog pred
        _x = rearrange(x, 'n t d -> (n t) d')
        hog_pred = self.hog_pred(_x)
        
        # optical flow
        prev_x = x[:,:-1, :]
        next_x = x[:,1:, :]
        diff_x = next_x - prev_x
        diff_x = rearrange(diff_x, 'n t d -> (n t) d')
        diff_x = self.to_spatial(diff_x)
        flow_pred = self.flow_pred(diff_x)
        
        return patch_pred, hog_pred, flow_pred