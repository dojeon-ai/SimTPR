import torch
import torch.nn as nn
import torch.nn.functional as F

INF = 1e9

class TemporalContrastiveLoss(nn.Module):
    # Temporal extension of SimCLR loss
    # Identical to the SimCLR loss if T=1

    def __init__(self, time_span, num_trajectory, temperature, device):
        super().__init__()
        self.time_span = time_span
        self.num_trajectory = num_trajectory
        self.temperature = temperature
        self.device = device

    def forward(self, z, done):
        # [params] z: (2*T*N, D)
        # [params] done: (T, N)
        T, N = self.time_span, self.num_trajectory
        
        # masking
        # logits_mask: exclude main diagonal in softmax
        logits_mask = torch.eye(z.shape[0], device=self.device)
        # temporal_mask: select log probability within the temporal window
        temporal_mask = torch.block_diag(*torch.ones(N, T, T, device=self.device))
        temporal_mask = temporal_mask.repeat(2,2)
        # done_mask: do not select the logits after done (different trajectory)
        done = done.float().T
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
        logits = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)
        logits = logits / self.temperature
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


class TemporalConsistencyLoss(nn.Module):
    # Temporal extension of BYOL loss
    # Identical to the BYOL loss if T=1

    def __init__(self):
        super().__init__()

    def forward(self, p, z):
        # T: trajectory_size, B: batch_size, D: representation_dim
        # p, z: (T, B, D)

        return loss

    



if __name__ == '__main__':
    print('[TEST LOSS Functions]')

