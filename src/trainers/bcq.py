from .base import BaseTrainer
from src.common.losses import ConsistencyLoss
from src.common.train_utils import LinearScheduler
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class BCQTrainer(BaseTrainer):
    name = 'bcq'
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

    def compute_loss(self, obs, act, rew, done, rtg):
        ##############
        # forward
        n, t, f, c, h, w = obs.shape
        # augmentation
        x = obs / 255.0
        x = rearrange(x, 'n t f c h w -> n (t f c) h w')
        x = self.aug_func(x)
        x = rearrange(x, 'n (t f c) h w -> n t f c h w', t=t, f=f, c=c)

        # online encoder
        y, _ = self.model.backbone(x)
        act_pred = self.model.head.act_predict(y)
        act_pred = rearrange(act_pred, 'n t d -> (n t) d')
        
        q_pred = self.model.head.q_predict(y)  
        q_pred = rearrange(q_pred, 'n t d -> (n t) d')
        q_pred = q_pred.gather(-1, act)
        
        # a'| (G(a'|s') / max_a G(a|s)) > τ
        inf = 1e9
        act_prob = F.softmax(act_pred, -1)
        with torch.no_grad():
            target_q_pred = self.target_model.head.q_predict(y)
            target_q_pred = rearrange(target_q_pred, 'n t d -> (n t) d')

            act_prob = act_prob / torch.max(act_prob, -1)[0].unsqueeze(-1)
            act_mask = (act_prob < self.cfg.tau) * inf
            target_q_pred = target_q_pred - act_mask
            target_q_pred = torch.max(target_q_pred, -1)[0].unsqueeze(-1)
        
        # loss
        # bc-loss
        act_loss_fn = nn.CrossEntropyLoss()
        act_ = rearrange(act, 'n t -> (n t)')
        act_loss = act_loss_fn(act_pred, act_)
        act_acc = torch.mean((torch.argmax(act_pred, 1) == act).float())

        # Bellman-loss
        bellman_loss_fn = nn.MSELoss()
        q_pred = rearrange(q_pred, 'n t -> (n t)')
        target_q_pred = rearrange(target_q_pred, 'n t -> (n t)')
        rew_ = rearrange(rew, 'n t -> (n t)')
        done_ = rearrange(done, 'n t -> (n t)').float()
        bellman_loss = bellman_loss_fn(q_pred + rew_, self.cfg.gamma * target_q_pred * (1-done_))
        bellman_loss = torch.mean(bellman_loss)
        
        loss = act_loss + bellman_loss
        
        ###############
        # logs        
        log_data = {'loss': loss.item(),
                    'act_loss': act_loss.item(),
                    'bellman_loss': bellman_loss.item(),
                    'act_acc': act_acc.item()}
        
        return loss, log_data

    