from .base import BaseTrainer
from src.common.losses import ConsistencyLoss, CURLLoss
from src.common.train_utils import LinearScheduler
from src.common.vit_utils import get_random_1d_mask
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import wandb

class BLTTrainer(BaseTrainer):
    name = 'blt'
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
        ####################
        # augmentation
        n, t, f, c, h, w = obs.shape
        
        # weak augmentation to both online & target
        x = obs / 255.0
        x = rearrange(x, 'n t f c h w -> n (t f c) h w')
        x1, x2 = self.aug_func(x), self.aug_func(x)
        x = torch.cat([x1, x2], axis=0)        
        x = rearrange(x, 'n (t f c) h w -> n t f c h w', t=t, f=f)
        act = torch.cat([act, act], axis=0)
        rew = torch.cat([rew, rew], axis=0)
        
        # mask the latent
        latent_shape = (n, t)
        mask_ratio = self.cfg.mask_ratio
        
        _, obs_mask, _ = get_random_1d_mask(latent_shape, mask_ratio)
        _, act_mask, _ = get_random_1d_mask(latent_shape, mask_ratio)
        _, rew_mask, _ = get_random_1d_mask(latent_shape, mask_ratio)
        obs_mask = torch.cat([obs_mask, obs_mask], axis=0).to(self.device)
        act_mask = torch.cat([act_mask, act_mask], axis=0).to(self.device)
        rew_mask = torch.cat([rew_mask, rew_mask], axis=0).to(self.device)
        
        #################
        # forward
        # online encoder
        y_o, _ = self.model.backbone(x)  
        
        dec_input = {
            'obs': y_o,
            'act': act,
            'rew': rew
        }
        
        dec_mask = {
            'obs': obs_mask,
            'act': act_mask,
            'rew': rew_mask
        }
    
        dec_output = self.model.head.decode(dec_input, dec_mask) 
        obs_o, act_o, rew_o = dec_output['obs'], dec_output['act'], dec_output['rew']   
        z_o = self.model.head.project(obs_o)
        p_o = self.model.head.predict(z_o)
        z1_o, z2_o = z_o.chunk(2)
        p1_o, p2_o = p_o.chunk(2)
        
        # target encoder
        with torch.no_grad():
            y_t, _ = self.target_model.backbone(x)
            z_t = self.target_model.head.project(y_t)
            z1_t, z2_t = z_t.chunk(2)

        #################
        # loss
        # observation loss (self-predictive)
        if self.cfg.obs_loss_type == 'consistency':
            obs_loss_fn = ConsistencyLoss(reduction='none')
        
        elif self.cfg.obs_loss_type == 'contrastive':
            obs_loss_fn = CURLLoss(self.cfg.temperature, reduction='none')
        
        p1_o, p2_o = rearrange(p1_o, 'n t d -> (n t) d'), rearrange(p2_o, 'n t d -> (n t) d')
        z1_t, z2_t = rearrange(z1_t, 'n t d -> (n t) d'), rearrange(z2_t, 'n t d -> (n t) d')
        obs_mask, _ = obs_mask.chunk(2)
        obs_mask = rearrange(obs_mask, 'n t -> (n t)')
        obs_loss = 0.5 * (obs_loss_fn(p1_o, z2_t) + obs_loss_fn(p2_o, z1_t))
        obs_loss = obs_loss * obs_mask
        obs_loss = torch.sum(obs_loss) / torch.sum(obs_mask)

        # action loss
        act_loss_fn = nn.CrossEntropyLoss(reduction='none')
        act_o = rearrange(act_o, 'n t d -> (n t) d')
        act_t = rearrange(act, 'n t -> (n t)')
        act_mask = rearrange(act_mask, 'n t -> (n t)')
        
        act_loss = act_loss_fn(act_o, act_t)
        act_loss = act_loss * act_mask
        act_loss = torch.sum(act_loss) / torch.sum(act_mask)
        
        act_acc = (torch.argmax(act_o, 1) == act_t).float()
        act_acc = act_acc * act_mask
        act_acc = torch.sum(act_acc) / torch.sum(act_mask)
        
        # reward loss
        rew_loss_fn = nn.MSELoss(reduction='none')
        rew_o = rearrange(rew_o, 'n t 1 -> (n t 1)')
        rew = rearrange(rew, 'n t -> (n t)')
        rew_mask = rearrange(rew_mask, 'n t -> (n t)')
        
        rew_loss = rew_loss_fn(rew_o, rew)
        rew_loss = rew_loss * rew_mask
        rew_loss = torch.sum(rew_loss) / torch.sum(rew_mask)

        loss = (self.cfg.obs_lmbda * obs_loss 
                + self.cfg.act_lmbda * act_loss 
                + self.cfg.rew_lmbda * rew_loss)
        
        ###############
        # logs
        # quantitative
        pos_idx = torch.eye(p1_o.shape[0], device=p1_o.device)
        sim = F.cosine_similarity(p1_o.unsqueeze(1), z2_t.unsqueeze(0), dim=-1)
        pos_sim = (torch.sum(sim * pos_idx) / torch.sum(pos_idx))
        neg_sim = (torch.sum(sim * (1-pos_idx)) / torch.sum(1-pos_idx))
        pos_neg_diff = pos_sim - neg_sim
        
        log_data = {'loss': loss.item(),
                    'obs_loss': obs_loss.item(),
                    'act_loss': act_loss.item(),
                    'rew_loss': rew_loss.item(),
                    'act_acc': act_acc.item(),
                    'pos_sim': pos_sim.item(),
                    'neg_sim': neg_sim.item(),
                    'pos_neg_diff': pos_neg_diff.item()}
        
        return loss, log_data

    def update(self, obs, act, rew, done, rtg):
        tau = self.tau_scheduler.get_value()
        for online, target in zip(self.model.parameters(), self.target_model.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data
