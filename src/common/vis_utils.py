import torch
import imageio
import os
import numpy as np
import wandb
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


class VideoRecorder:
    def __init__(self,
                 save_dir,
                 render_size=256,
                 fps=20,
                 camera_id=0,
                 use_wandb=False):
        if save_dir is not None:
            self.save_dir = save_dir
            self.save_dir.mkdir(exist_ok=False, parents=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.camera_id = camera_id
        self.use_wandb = use_wandb

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            if hasattr(env, 'physics'):
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=self.camera_id)
            else:
                frame = env.render()
            self.frames.append(frame)

    def log_to_wandb(self):
        frames = np.transpose(np.array(self.frames), (0, 3, 1, 2))
        fps, skip = 6, 8
        wandb.log({
            'eval/video':
            wandb.Video(frames[::skip, :, ::2, ::2], fps=fps, format="gif")
        })

    def save(self, file_name):
        if self.enabled:
            if self.use_wandb:
                self.log_to_wandb()
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)