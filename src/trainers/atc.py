from .base import BaseTrainer
from src.common.losses import CURLLoss
from src.common.train_utils import LinearScheduler
from einops import rearrange
import torch
import torch.nn.functional as F
import copy

class ATCTrainer(BaseTrainer):
    name = 'atc'
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
        self.target_model = copy.deepcopy(self.model).to(self.device)  
        update_steps = len(self.train_loader) * self.cfg.num_epochs
        cfg.tau_scheduler.step_size = update_steps
        self.tau_scheduler = LinearScheduler(**cfg.tau_scheduler)

    def compute_loss(self, obs, act, rew, done, rtg, mode):
        #################
        # forward
        n, t, f, c, h, w = obs.shape
        # augmentation
        x = obs / 255.0
        x = rearrange(x, 'n t f c h w -> n (t f c) h w')
        x1, x2 = self.aug_func(x), self.aug_func(x)
        x1 = rearrange(x1, 'n (t f c) h w -> n t f c h w', t=t, f=f)
        x2 = rearrange(x2, 'n (t f c) h w -> n t f c h w', t=t, f=f)
        x = torch.cat([x1, x2], axis=0)
        x_o = x[:, :-self.cfg.k_step]
        x_t = x[:, self.cfg.k_step:]
        
        # online encoder
        y_o, _ = self.model.backbone(x_o)   
        z_o = self.model.head.project(y_o)
        p_o = self.model.head.predict(z_o)
        p_o = rearrange(p_o, 'n t d -> (n t) d')
        p1_o, p2_o = p_o.chunk(2)

        # target encoder
        with torch.no_grad():
            y_t, _ = self.target_model.backbone(x_t)
            z_t = self.target_model.head.project(y_t)
            z_t = rearrange(z_t, 'n t d -> (n t) d')
            z1_t, z2_t = z_t.chunk(2)

        # loss
        loss_fn = CURLLoss(temperature=self.cfg.temperature)
        loss = 0.5 * (loss_fn(p1_o, z2_t) + loss_fn(p2_o, z1_t))
        loss = torch.mean(loss)

        ###############
        # logs
        pos_idx = torch.eye(p1_o.shape[0], device=p1_o.device)
        sim = F.cosine_similarity(p1_o.unsqueeze(1), z2_t.unsqueeze(0), dim=-1)
        pos_sim = (torch.sum(sim * pos_idx) / torch.sum(pos_idx))
        neg_sim = (torch.sum(sim * (1-pos_idx)) / torch.sum(1-pos_idx))
        pos_neg_diff = pos_sim - neg_sim
        
        log_data = {'loss': loss.item(),
                    'pos_sim': pos_sim.item(),
                    'neg_sim': neg_sim.item(),
                    'pos_neg_diff': pos_neg_diff.item()}
        
        return loss, log_data
    
    def update(self, obs, act, rew, done, rtg):
        tau = self.tau_scheduler.get_value()
        for online, target in zip(self.model.parameters(), self.target_model.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data
    