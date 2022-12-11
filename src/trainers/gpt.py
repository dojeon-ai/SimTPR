from .base import BaseTrainer
from src.common.losses import ConsistencyLoss, CURLLoss, BarlowLoss
from src.common.train_utils import LinearScheduler
from src.common.vit_utils import get_random_3d_mask, get_3d_masked_input, restore_masked_input
from src.common.beam import Beam
from einops import rearrange
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
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
        max_rtg = 0
        datasets = self.train_loader.dataset.datasets
        for dataset in datasets:
            max_rtg = max(max_rtg, torch.max(dataset.rtg).item())
        self.max_rtg = max_rtg
        
        
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
        obs_p1, obs_p2 = obs_p.chunk(2)
        obs_p1, obs_p2 = rearrange(obs_p1, 'n t d -> (n t) d'), rearrange(obs_p2, 'n t d -> (n t) d')

        obs_t1, obs_t2 = obs_t.chunk(2)
        obs_t1, obs_t2 = rearrange(obs_t1, 'n t d -> (n t) d'), rearrange(obs_t2, 'n t d -> (n t) d')

        cons_loss_fn = ConsistencyLoss()          
        cons_loss = 0.5 * (cons_loss_fn(obs_p1, obs_t2.detach()) + cons_loss_fn(obs_p2, obs_t1.detach()))
        cons_loss = torch.mean(cons_loss)
        
        barlow_loss_fn = BarlowLoss(self.cfg.lmbda)
        t1 = F.normalize(obs_t1, dim=-1, p=2)
        t2 = F.normalize(obs_t2, dim=-1, p=2)
        barlow_loss = barlow_loss_fn(t1, t2)
                    
        obs_loss = self.cfg.cons_lmbda * cons_loss + self.cfg.barlow_lmbda * barlow_loss
        
            
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
                self.cfg.act_lmbda * act_loss)
        
        ###############
        # logs
        # quantitative
        if train:
            log_data = {'loss': loss.item(),
                        'obs_loss': obs_loss.item(),
                        'barlow_loss': barlow_loss.item(),
                        'cons_loss': cons_loss.item(),
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

            
    def evaluate_policy(self):
        self.model.eval()
        
        # initialize history
        C = self.cfg.rollout.context_len
        obs_list = deque(maxlen=C)
        act_list = deque(maxlen=C) 
        rew_list = deque(maxlen=C)
        rtg_list = deque(maxlen=C)

        # initialize tree
        rollout_cfg = self.cfg.rollout
        rollout_type = rollout_cfg.pop('type')

        if rollout_type == 'mcts':        
            raise NotImplemented

        elif rollout_type == 'beam':
            tree = Beam(model=self.model, 
                        device=self.device,
                        num_envs=self.cfg.num_envs,
                        action_size=self.cfg.action_size,
                        rtg_scale=self.max_rtg,
                        **rollout_cfg)

        # run trajectory
        obs = self.env.reset()
        rtg = np.array([self.max_rtg] * self.cfg.num_envs)
        while True:
            # encode obs
            obs = np.array(obs).astype(np.float32)
            obs = obs / 255.0
            obs = torch.FloatTensor(obs).to(self.device)
            obs = rearrange(obs, 'n f c h w -> n 1 f c h w')
            with torch.no_grad():
                obs, _ = self.model.backbone(obs)
                obs = self.model.head.obs_to_latent(obs)
                obs = obs.cpu()

            obs_list.append(obs)

            # select action with rollout
            state = {
                'obs_list': obs_list,
                'act_list': act_list,
                'rew_list': rew_list,
                'rtg_list': rtg_list
            }

            beam = tree.rollout(state)            
            tree_action = beam['act_batch'].numpy()[:, 0]
            rand_action = np.random.randint(0, self.cfg.action_size-1, self.cfg.num_envs)

            # eps-greedy
            eps = self.cfg.eval_eps
            prob = np.random.rand(self.cfg.num_envs)
            rand_idx = (prob <= eps)
            action = rand_idx * rand_action + (1-rand_idx) * tree_action

            # step
            next_obs, reward, done, info = self.env.step(action)

            # logger
            self.agent_logger.step(obs, reward, done, info)

            # move on
            if np.sum(self.agent_logger.traj_done) == self.cfg.num_envs:
                break
            else:
                obs = next_obs
                act_list.append(action)
                rew_list.append(reward)
                rtg_list.append(rtg / self.max_rtg)
                rtg = rtg - reward
    
        log_data = self.agent_logger.fetch_log()
        
        return log_data

            