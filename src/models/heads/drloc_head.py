import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseHead


def randn_sampling(maxint, sample_size, device):
    return torch.randint(maxint, size=(sample_size,), device=device)


def collect_samples(feats, pxy, batch_size):
    return torch.stack([feats[i, :, pxy[i][:,0], pxy[i][:,1]] for i in range(batch_size)], dim=0)


def collect_samples_faster(feats, pxy, batch_size):
    n,c,h,w = feats.size()
    feats = feats.view(n, c, -1).permute(1,0,2).reshape(c, -1)  # [n, c, h, w] -> [n, c, hw] -> [c, nhw]
    pxy = ((torch.arange(n).long().to(pxy.device) * h * w).view(n, 1) + pxy[:,:,0]*h + pxy[:,:,1]).view(-1)  # [n, m, 2] -> [nm]
    return (feats[:,pxy]).view(c, n, -1).permute(1,0,2)


class DRLocHead(BaseHead):
    name = 'drloc'
    def __init__(self, 
                 in_features,
                 hid_features,
                 out_features,
                 sample_size):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hid_features),
            nn.ReLU(),
            nn.Linear(in_features=hid_features, out_features=out_features)
        )
        self.sample_size = sample_size

    def forward(self, x):
        N, T, H, W, D = x.size()
        S = self.sample_size
        pz1, pz2 = randn_sampling(T, S*N, x.device), randn_sampling(T, S*N, x.device)
        px1, px2 = randn_sampling(H, S*N, x.device), randn_sampling(H, S*N, x.device)
        py1, py2 = randn_sampling(W, S*N, x.device), randn_sampling(W, S*N, x.device)
        
        # get prediction
        batch_idx = torch.arange(N, device=x.device).repeat(S)
        e1 = x[batch_idx, pz1, px1, py1]
        e2 = x[batch_idx, pz2, px2, py2]
        x = torch.cat([e1, e2], dim=-1)
        pred = self.predictor(x)
        
        # get target
        tz = torch.abs(pz1-pz2) / T
        tx = torch.abs(px1-px2) / H
        ty = torch.abs(py1-py2) / W
        target = torch.cat([tz.unsqueeze(-1), tx.unsqueeze(-1), ty.unsqueeze(-1)], dim=-1)

        return pred, target
        