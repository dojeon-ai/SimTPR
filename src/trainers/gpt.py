from .base import BaseTrainer
from src.common.losses import ConsistencyLoss, CURLLoss, BarlowLoss
from src.common.train_utils import LinearScheduler
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import random
import wandb

class GPTTrainer(BaseTrainer):
    name = 'gpt'
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
        
        
    def compute_loss(self, obs, act, rew, done, rtg, train=True):
        ####################
        # augmentation
        n, t, f, c, h, w = obs.shape
        x = obs / 255.0
        x = rearrange(x, 'n t f c h w -> n (t f c) h w')
        x1, x2 = self.aug_func(x), self.aug_func(x)
        x = torch.cat([x1, x2], axis=0)        
        x = rearrange(x, 'n (t f c) h w -> n t f c h w', t=t, f=f)
        act = torch.cat([act, act], axis=0)
        
        #################
        # forward
        
        # encode
        y, _ = self.model.backbone(x)  
        z = self.model.head.encode_obs(y)
        
        obs_o = z[:, :-1]
        obs_t = z[:, 1:]
        act = act[:, :-1]
        
        # decode
        d_type = self.cfg.dataset_type
        obs_d, act_d = self.model.head.decode(obs_o, act, d_type)  
        obs_p, act_p = self.model.head.predict(obs_d, act_d) 

        #################
        # loss
        
        # obs loss            
        obs_t1, obs_t2 = obs_t.chunk(2)
        obs_t1, obs_t2 = rearrange(obs_t1, 'n t d -> (n t) d'), rearrange(obs_t2, 'n t d -> (n t) d')

        if self.cfg.loss_type == 'cons':
            obs_loss_fn = ConsistencyLoss() 
            obs_p1, obs_p2 = obs_p.chunk(2)
            obs_p1, obs_p2 = rearrange(obs_p1, 'n t d -> (n t) d'), rearrange(obs_p2, 'n t d -> (n t) d')
            obs_loss = 0.5 * (obs_loss_fn(obs_p1, obs_t2.detach()) + obs_loss_fn(obs_p2, obs_t1.detach()))
            obs_loss = torch.mean(obs_loss)
            
        elif self.cfg.loss_type == 'cont':
            obs_loss_fn = CURLLoss(self.cfg.temperature) 
            obs_d1, obs_d2 = obs_d.chunk(2)
            obs_d1, obs_d2 = rearrange(obs_d1, 'n t d -> (n t) d'), rearrange(obs_d2, 'n t d -> (n t) d')
            obs_loss = 0.5 * (obs_loss_fn(obs_d1, obs_t2) + obs_loss_fn(obs_d2, obs_t1))
            obs_loss = torch.mean(obs_loss)
            
        else:
            raise NotImplemented
        
        reg_loss_fn = BarlowLoss(self.cfg.barlow_lmbda)
        t1 = F.normalize(obs_t1, dim=-1, p=2)
        t2 = F.normalize(obs_t2, dim=-1, p=2)
        reg_loss = reg_loss_fn(t1, t2)
            
        # act loss
        act_loss_fn = nn.CrossEntropyLoss(reduction='none')
        act_p = rearrange(act_p, 'n t d -> (n t) d')
        act_t = rearrange(act, 'n t -> (n t)')
        act_loss = act_loss_fn(act_p, act_t)        
        act_loss = torch.mean(act_loss)
        act_acc = torch.mean((torch.argmax(act_p, 1) == act_t).float())
        
        if d_type == 'video':
            act_loss = torch.Tensor([0.0]).to(x.device)
        
        loss = (self.cfg.obs_lmbda * obs_loss + 
                self.cfg.act_lmbda * act_loss + 
                self.cfg.reg_lmbda * reg_loss)
        
        ###############
        # logs
        # quantitative
        if train:
            log_data = {'loss': loss.item(),
                        'obs_loss': obs_loss.item(),
                        'reg_loss': reg_loss.item(),
                        'act_loss': act_loss.item(),
                        'act_acc': act_acc.item()}
            
        else:
            pos_idx = torch.eye(obs_p1.shape[0], device=x.device)
            sim = F.cosine_similarity(obs_p1.unsqueeze(1), obs_t2.unsqueeze(0), dim=-1)
            pos_sim = (torch.sum(sim * pos_idx) / torch.sum(pos_idx))
            neg_sim = (torch.sum(sim * (1-pos_idx)) / torch.sum(1-pos_idx))
            pos_neg_diff = pos_sim - neg_sim

            s = torch.linalg.svdvals(obs_t1)
            rank_eps001 = torch.sum(s > 0.01)
            rank_eps01 = torch.sum(s > 0.1)
            rank_eps1 = torch.sum(s > 1)
        
            log_data = {'pos_sim': pos_sim.item(),
                        'neg_sim': neg_sim.item(),
                        'pos_neg_diff': pos_neg_diff.item(),
                        'rank_eps001': rank_eps001,
                        'rank_eps01': rank_eps01,
                        'rank_eps1': rank_eps1}
        
        return loss, log_data

    
    def update(self, obs, act, rew, done, rtg):
        pass

