import torch
from einops import rearrange, repeat

# attention rollout
# https://jacobgil.github.io/deeplearning/vision-transformer-explainability

def rollout_attn_maps(attn_maps):
    """
    [param]: attn_maps: list of L attn_map:(N, H, T*(P+1), T*(P+1))
    L=num_layers, N=batch_size, H=num_heads, T=t_step, P=num_patches
    """
    attn_maps = torch.stack(attn_maps)
    
    # average the attention weights across all heads
    attn_maps = torch.mean(attn_maps, dim=2)
    
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights for layer-norm
    L, N, TP, TP = attn_maps.shape
    residual = torch.eye(TP, device=attn_maps.device)
    attn_maps = attn_maps + residual
    attn_maps = attn_maps / attn_maps.sum(dim=-1).unsqueeze(-1)
    
    # Recursivley multiply the weight matrices
    joint_attn_maps = torch.zeros_like(attn_maps)
    joint_attn_maps[0] = attn_maps[0]
    
    for l in range(1, attn_maps.shape[0]):
        joint_attn_maps[l] = torch.matmul(attn_maps[l], joint_attn_maps[l-1])
    
    # joint attn_map from the last layer (N, T*(P+1), T*(P+1))
    attn_maps = joint_attn_maps[-1]
    
    return attn_maps
