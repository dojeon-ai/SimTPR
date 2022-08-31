import torch
import torch.nn as nn
import torch.nn.functional as F
from munch import Munch

def randn_sampling(maxint, sample_size, batch_size):
    return torch.randint(maxint, size=(batch_size, sample_size, 2))

def collect_samples(feats, pxy, batch_size):
    return torch.stack([feats[i, :, pxy[i][:,0], pxy[i][:,1]] for i in range(batch_size)], dim=0)

def collect_samples_faster(feats, pxy, batch_size):
    n,c,h,w = feats.size()
    feats = feats.view(n, c, -1).permute(1,0,2).reshape(c, -1)  # [n, c, h, w] -> [n, c, hw] -> [c, nhw]
    pxy = ((torch.arange(n).long().to(pxy.device) * h * w).view(n, 1) + pxy[:,:,0]*h + pxy[:,:,1]).view(-1)  # [n, m, 2] -> [nm]
    return (feats[:,pxy]).view(c, n, -1).permute(1,0,2)


class DenseRelativeLoc(nn.Module):
    def __init__(self, in_dim, out_dim=3, sample_size=64, use_abs=False):
        super(DenseRelativeLoc, self).__init__()
        self.sample_size = sample_size
        self.drloc_mode = drloc_mode
        self.in_dim  = in_dim
        self.use_abs = use_abs

        if self.drloc_mode == "l1":
            self.out_dim = out_dim
            self.layers = nn.Sequential(
                nn.Linear(in_dim*2, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, self.out_dim)
            )
            
    def forward_features(self, x):
        # x, feature map with shape: [B, C, H, W]
        B, C, H, W = x.size()

        pxs = randn_sampling(H, self.sample_size, B).detach()
        pys = randn_sampling(H, self.sample_size, B).detach()

        deltaxy = (pxs-pys).float().to(x.device) # [B, sample_size, 2]

        ptsx = collect_samples_faster(x, pxs, B).transpose(1,2).contiguous() # [B, sample_size, C]
        ptsy = collect_samples_faster(x, pys, B).transpose(1,2).contiguous() # [B, sample_size, C]

        pred_feats = self.layers(torch.cat([ptsx, ptsy], dim=2))
        return pred_feats, deltaxy, H

    
    def forward(self, x, normalize=False):
        pred_feats, deltaxy, H = self.forward_features(x)
        deltaxy = deltaxy.view(-1, 2) # [B*sample_size, 2]
        deltaxy += (H-1)
        if normalize:
            deltaxy /= float(2*(H - 1))
        
        predxy = pred_feats.view(-1, self.out_dim) # [B*sample_size, Output_size]
 
        return predxy, deltaxy