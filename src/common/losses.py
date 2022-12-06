import torch
import torch.nn as nn
import torch.nn.functional as F

INF = 1e9

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
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
        loss = -mean_log_prob
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss
    
    
class CURLLoss(nn.Module):
    def __init__(self, temperature, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
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
        loss = -mean_log_prob
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss

    
class ConsistencyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, p, z):
        p = F.normalize(p, dim=-1, p=2)
        z = F.normalize(z, dim=-1, p=2)
        
        loss = 2 - 2 * (p * z).sum(dim=-1)
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss
        
        
class BarlowLoss(nn.Module):
    def __init__(self, lmbda, reduction='mean'):
        super().__init__()
        self.lmbda = lmbda
        self.reduction = reduction
        
    def _off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        
    def forward(self, z1, z2):
        n, d = z1.shape
        # normalize along batch dim
        z1 = (z1 - z1.mean(0)) / z1.std(0) # NxD
        z2 = (z2 - z2.mean(0)) / z2.std(0) # NxD
        
        # cross correltation matrix
        cor = torch.mm(z1.T, z2)
        cor.div_(n)
        
        # loss
        on_diag = torch.diagonal(cor).add_(-1).pow_(2).sum()
        off_diag = self._off_diagonal(cor).pow_(2).sum()
        
        loss = on_diag + self.lmbda * off_diag
        
        if self.reduction == 'mean':
            return loss
        else:
            raise ValueError
    

############################################
# loss function for linear probing
class SoftmaxFocalLoss(nn.Module):
    """Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self, alpha=None, gamma: float = 0.0, reduction: str = "mean", ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ("mean", "sum", "none"):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(weight=alpha, reduction="none", ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ["alpha", "gamma", "ignore_index", "reduction"]
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f"{k}={v}" for k, v in zip(arg_keys, arg_vals)]
        arg_str = ", ".join(arg_strs)
        return f"{type(self).__name__}({arg_str})"

    def forward(self, x, y):
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return 0.0
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

