from .base import BaseTrainer
import torch
import torch.nn.functional as F
from torch.nn.functional import mse_loss
from einops import rearrange
import wandb


class VAETrainer(BaseTrainer):
    name = 'vae'
    def __init__(self,
                 cfg,
                 device,
                 train_loader,
                 eval_act_loader,
                 eval_rew_loader,
                 env,
                 logger, 
                 agent_logger,
                 aug_func,
                 model):
        
        super().__init__(cfg, device, 
                         train_loader, eval_act_loader, eval_rew_loader, env,
                         logger, agent_logger, aug_func, model)  

        self.use_stack_out = self.cfg.use_stack_out

    def compute_loss(self, obs, act, rew, done, rtg, mode):
        n, t, f, c, h, w = obs.shape
        # augmentation
        x = obs / 255.0
        x = rearrange(x, 'n t f c h w -> n (t f c) h w')
        x = self.aug_func(x)        
        x = rearrange(x, 'n (t f c) h w -> n t f c h w', t=t, f=f)

        # encoder: (n, t, f, c, h, w) -> (n, t, d)
        y, _ = self.model.backbone(x)

        # mu, log_var: (n t hid_dim)
        mu, log_var = self.model.head.encode(y)
        z = self.model.head.reparameterize(mu, log_var)
        recon = self.model.head.decode(z)
        recon = rearrange(recon, '(n t) (f c) h w -> n t f c h w', n=n, t=t, f=f, c=c)

        # compute loss
        obs_target = obs / 255.0
        recon_loss = mse_loss(recon, obs_target, reduction='none').mean([0, 1, 2]).sum()
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = -1))

        kl_lmbda = self.cfg.kl_lmbda
        loss = recon_loss + kl_lmbda * kl_loss

        # logs
        # get reconstructed image
        obs_target = rearrange(obs_target, 'n t f c h w -> n t (f c) h w')[0, :, -1].unsqueeze(1)
        recon = rearrange(recon, 'n t f c h w -> n t (f c) h w')[0, :, -1].unsqueeze(1)
        recon = recon.to(float)
        recon = torch.where(recon >= 1.0, 1.0, recon)
        recon = torch.where(recon < 0.0, 0.0, recon)

        log_data = {'loss': loss.item(),
                    'recon_loss': recon_loss.item(),
                    'kl_loss': kl_loss.item()
                    #'recon_image': wandb.Image(recon),
                    #'obs_image': wandb.Image(obs_target)
        }
            
        return loss, log_data
            