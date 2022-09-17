import torch
import torch.nn as nn
import torch.nn.functional as F

INF = 1e9

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z):
        """
        [params] z: torch.Tensor (2*n, d)
        """
        # cosine_sim: identical to matmul in l2-normalized space
        logits = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)
        logits = logits / self.temperature
        
        # mask self-similarity
        self_mask = torch.eye(z.shape[0], device=z.device)
        logits = logits - self_mask * INF
        
        # compute log-likelihood 
        exp_logits = torch.exp(logits)
        log_probs = logits - torch.log(torch.sum(exp_logits, 1, keepdim=True))
        
        # get log-likelihood of an augmented view
        n = z.shape[0] // 2
        pos_mask = torch.eye(n, device=z.device)
        pos_mask = pos_mask.repeat(2,2)
        aug_idx = pos_mask - self_mask
        log_probs = aug_idx * log_probs
        
        # compute loss
        mean_log_prob = torch.sum(log_probs, 1) / torch.sum(aug_idx,1)
        loss = -torch.mean(mean_log_prob)
        
        return loss   
    
    
class CURLLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, p, z):
        """
        [params] p: torch.Tensor (n, d)
        [params] z: torch.Tensor (n, d)
        """
        # cosine_sim: identical to matmul in l2-normalized space
        logits = F.cosine_similarity(p.unsqueeze(1), z.unsqueeze(0), dim=-1)
        logits = logits / self.temperature
        
        # compute log-likelihood 
        exp_logits = torch.exp(logits)
        log_probs = logits - torch.log(torch.sum(exp_logits, 1, keepdim=True))
        
        # get log-likelihood of an augmented view
        aug_idx = torch.eye(p.shape[0], device=p.device)
        log_probs = aug_idx * log_probs
        
        # compute loss
        mean_log_prob = torch.sum(log_probs, 1) / torch.sum(aug_idx,1)
        loss = -torch.mean(mean_log_prob)
        
        return loss   

    
class ConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, p, z):
        p = F.normalize(p, dim=-1, p=2)
        z = F.normalize(z, dim=-1, p=2)
        
        return 2 - 2 * (p * z).sum(dim=-1)

