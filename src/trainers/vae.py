from .base import BaseTrainer
from src.common.losses import BarlowLoss
import torch
import torch.nn.functional as F
from einops import rearrange


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

        # encoder
        # output: n t d
        y, _ = self.model.backbone(x)

        # vae
        # mu, log_var: n t hid_dim
        mu, log_var = self.model.head.encode(y)
        z = self.model.head.reparameterize(mu, log_var)
        recon = self.model.head.decode(z)

        recon = rearrange(recon, '(n t) (f c) h w -> n t f c h w', n=n, t=t, f=f, c=c)

        result = self.model.head.loss_function(recon, obs / 255.0, mu, log_var)

        loss = result['loss']

        log_data = {x:y if x !='loss' else y.detach() for x, y in result.items()}

        return loss, log_data