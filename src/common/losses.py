import torch
import torch.nn as nn
import torch.nn.functional as F

INF = 1e9

class TemporalContrastiveLoss(nn.Module):
    # Temporal extension of SimCLR loss
    # Identical to the SimCLR loss if T=1
    def __init__(self, num_trajectory, t_step, temperature, device):
        super().__init__()
        self.num_trajectory = num_trajectory
        self.t_step = t_step
        self.temperature = temperature
        self.device = device

    def forward(self, z, done):
        # [params] z: (2*N*T, D)
        # [params] done: (N, T)
        N, T = self.num_trajectory, self.t_step
        
        # masking
        # logits_mask: exclude main diagonal in softmax
        logits_mask = torch.eye(z.shape[0], device=self.device)
        # temporal_mask: select log probability within the temporal window
        temporal_mask = torch.block_diag(*torch.ones(N, T, T, device=self.device))
        temporal_mask = temporal_mask.repeat(2,2)
        # done_mask: do not select the logits after done (different trajectory)
        done = done.float()
        done_idx = torch.nonzero(done==1)
        for idx in done_idx:
            row = idx[0]
            col = idx[1]
            done[row, col] = 0
            done[row, col+1:] = 1
        done_mask = 1 - done.flatten().float()
        done_mask = torch.mm(done_mask.unsqueeze(1), done_mask.unsqueeze(0))
        done_mask = done_mask * torch.block_diag(*torch.ones(N,T,T, device=self.device))
        done_mask = done_mask.repeat(2,2)
        positive_mask = (1-done_mask) * (1-temporal_mask) + logits_mask

        # Get log_probs within temporal window
        # cosine_sim: identical to matmul in l2-normalized space
        _logits = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)
        logits = _logits / self.temperature
        # mask self-similarity
        logits = logits - logits_mask * INF
        exp_logits = torch.exp(logits)
        log_probs = logits - torch.log(torch.sum(exp_logits, 1, keepdim=True))
        # compute mean of log-likelihood over temporal window
        # supclr_out: https://arxiv.org/pdf/2004.11362.pdf
        log_probs = (1-positive_mask) * log_probs
        mean_log_prob = torch.sum(log_probs, 1) / torch.sum((1-positive_mask),1)
        
        # compute loss
        loss = -torch.mean(mean_log_prob)
        
        return loss

class TemporalCURLLoss(nn.Module):
    # Temporal extension of CURL loss
    # Identical to the CURL loss if T=1
    def __init__(self, num_trajectory, t_step, temperature, device):
        super().__init__()
        self.num_trajectory = num_trajectory
        self.t_step = t_step
        self.temperature = temperature
        self.device = device

    def forward(self, p, z, done):
        # [params] p, z (N*T, D)
        # [params] done: (N, T)
        N, T = self.num_trajectory, self.t_step
        
        # masking
        # temporal_mask: select log probability within the temporal window
        temporal_mask = torch.block_diag(*torch.ones(N, T, T, device=self.device))
        # done_mask: do not select the logits after done (different trajectory)
        done = done.float()
        done_idx = torch.nonzero(done==1)
        for idx in done_idx:
            row = idx[0]
            col = idx[1]
            done[row, col] = 0
            done[row, col+1:] = 1
        done_mask = 1 - done.flatten().float()
        done_mask = torch.mm(done_mask.unsqueeze(1), done_mask.unsqueeze(0))
        done_mask = done_mask * torch.block_diag(*torch.ones(N,T,T, device=self.device))
        positive_mask = (1-done_mask) * (1-temporal_mask)
        
        # Get log_probs within temporal window
        # cosine_sim: identical to matmul in l2-normalized space
        _logits = F.cosine_similarity(p.unsqueeze(1), z.unsqueeze(0), dim=-1)
        logits = _logits / self.temperature
        exp_logits = torch.exp(logits)
        log_probs = logits - torch.log(torch.sum(exp_logits, 1, keepdim=True))
        # compute mean of log-likelihood over temporal window
        # supclr_out: https://arxiv.org/pdf/2004.11362.pdf
        log_probs = (1-positive_mask) * log_probs
        mean_log_prob = torch.sum(log_probs, 1) / torch.sum((1-positive_mask),1)
        
        # compute loss
        loss = -torch.mean(mean_log_prob)
        
        return loss


class TemporalConsistencyLoss(nn.Module):
    # Temporal extension of BYOL loss
    # Identical to the BYOL loss if T=1
    def __init__(self, num_trajectory, t_step, device):
        super().__init__()
        self.num_trajectory = num_trajectory
        self.t_step = t_step
        self.device = device
        
    def forward(self, p, z, done):
        # T: trajectory_size, N: batch_size, D: representation_dim
        # [params] p, z: (N*T, D)
        # [params] done: (N, T)
        N, T = self.num_trajectory, self.t_step

        # logits: (N, T, T)
        logits = F.cosine_similarity(p.unsqueeze(1), z.unsqueeze(0), dim=-1)
        
        # temporal_mask: select log probability within the temporal window
        temporal_mask = torch.block_diag(*torch.ones(N, T, T, device=self.device))
        # done_mask: do not select the logits after done (different trajectory)
        done = done.float()
        done_idx = torch.nonzero(done==1)
        for idx in done_idx:
            row = idx[0]
            col = idx[1]
            done[row, col] = 0
            done[row, col+1:] = 1
        done_mask = 1 - done.flatten().float()
        done_mask = torch.mm(done_mask.unsqueeze(1), done_mask.unsqueeze(0))
        done_mask = done_mask * torch.block_diag(*torch.ones(N,T,T, device=self.device))
        positive_mask = (1-done_mask) * (1-temporal_mask)

        # compute mean of the similarity
        similarity = (1-positive_mask) * logits
        similarity = torch.sum(similarity, 1) / torch.sum((1-positive_mask),1)
        loss = -torch.mean(similarity)
        
        return loss
    
    
class TemporalSimilarityLoss(nn.Module):
    def __init__(self, num_trajectory, t_step, device):
        super().__init__()
        self.num_trajectory = num_trajectory
        self.t_step = t_step
        self.device = device
        
    def forward(self, z, done):
        # [params] z: (2*N*T, D)
        # [params] done: (N, T)
        N, T = self.num_trajectory, self.t_step
        
        # masking
        # logits_mask: exclude main diagonal in softmax
        logits_mask = torch.eye(z.shape[0], device=self.device)
        # temporal_mask: select log probability within the temporal window
        temporal_mask = torch.block_diag(*torch.ones(N, T, T, device=self.device))
        temporal_mask = temporal_mask.repeat(2,2)
        # done_mask: do not select the logits after done (different trajectory)
        done = done.float()
        done_idx = torch.nonzero(done==1)
        for idx in done_idx:
            row = idx[0]
            col = idx[1]
            done[row, col] = 0
            done[row, col+1:] = 1
        done_mask = 1 - done.flatten().float()
        done_mask = torch.mm(done_mask.unsqueeze(1), done_mask.unsqueeze(0))
        done_mask = done_mask * torch.block_diag(*torch.ones(N,T,T, device=self.device))
        done_mask = done_mask.repeat(2,2)
        positive_mask = (1-done_mask) * (1-temporal_mask) + logits_mask

        # Get log_probs within temporal window
        # cosine_sim: identical to matmul in l2-normalized space
        _logits = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)
        
        # logs
        positive_idx = torch.nonzero(1-positive_mask, as_tuple=True)
        positive_sim = torch.mean(_logits[positive_idx])
        negative_idx = torch.nonzero(positive_mask - logits_mask, as_tuple=True)
        negative_sim = torch.mean(_logits[negative_idx])
        
        return positive_sim, negative_sim



if __name__ == '__main__':
    print('[TEST Loss Functions]')
    N, T, D = 2, 3, 2
    device = torch.device('cuda:0')
    z1 = torch.FloatTensor([[[0.0, 1.0],
                             [1.0, 0.0],
                             [0.707, 0.707]],
                            
                            [[-1.0, 0.0],
                             [-1.0, 0.0],
                             [-0.707, 0.707]],]).to(device) #.permute(1,0,2)

    z2 = torch.FloatTensor([[[1.0, 0.0],
                             [0.707, -0.707],
                             [0.707, 0.707]],
                            
                            [[-1.0, 0.0],
                             [-0.707, 0.707],
                             [-1.0, 0.0]],]).to(device) #.permute(1,0,2)

    z1 = z1.reshape(N*T, D)
    z2 = z2.reshape(N*T, D)
    z = torch.cat([z1, z2])

    print('[1. TEST Temporal Contrastive Loss]')
    print('[1.1 TEST loss without done]')
    loss_fn = TemporalContrastiveLoss(num_trajectory=N, 
                                      t_step=T, 
                                      temperature=1.0, 
                                      device=device)
    done = torch.zeros((2,3)).to(device)
    loss = loss_fn(z, done)
    
    print('[2. TEST Temporal Consistency Loss]')
    loss_fn = TemporalConsistencyLoss(num_trajectory=N, 
                                      t_step=T, 
                                      device=device)
    done = torch.zeros((2,3)).to(device)
    loss = loss_fn(z1, z2, done)
    import pdb
    pdb.set_trace()