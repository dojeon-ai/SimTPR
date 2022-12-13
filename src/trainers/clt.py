from .base import BaseTrainer
from src.common.losses import ConsistencyLoss, CURLLoss
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

class CLTTrainer(BaseTrainer):
    name = 'clt'
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
        
        max_rtg = 0
        datasets = self.train_loader.dataset.datasets
        for dataset in datasets:
            max_rtg = max(max_rtg, torch.max(dataset.rtg).item())
        self.max_rtg = max_rtg
        
        
    def compute_loss(self, obs, act, rew, done, rtg, mode):
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
        rtg = torch.cat([rtg, rtg], axis=0) / self.max_rtg

        # strong augmentation to online: (masking)
        assert self.cfg.mask_type in {'none', 'pixel'}
        if self.cfg.mask_type == 'none':
            pass
        
        elif self.cfg.mask_type == 'pixel':
            ph, pw = self.cfg.patch_size[0], self.cfg.patch_size[1]
            nh, nw = (h // ph), (w//pw)
            np = nh * nw
            
            # generate random-mask
            video_shape = (2*n, t, np)
            ids_keep, _, ids_restore = get_random_3d_mask(video_shape, 
                                                          self.cfg.mask_ratio,
                                                          self.cfg.mask_strategy)
            ids_keep = ids_keep.to(x.device)
            ids_restore = ids_restore.to(x.device)
            
            # mask & restore
            x_o = rearrange(x, 'n t f c (nh ph) (nw pw) -> n t (nh nw) (ph pw f c)', ph=ph, pw=pw)
            x_o = get_3d_masked_input(x_o, ids_keep, self.cfg.mask_strategy)
            x_o = restore_masked_input(x_o, ids_restore)
            x_o = rearrange(x_o, 'n (t nh nw) (ph pw f c) -> n t f c (nh ph) (nw pw)', 
                            t=t, f=f, c=c, nh=nh, nw=nw, ph=ph, pw=pw)
            
        x_o = x[:, :-1]
        x_t = x[:, 1:]
        act = act[:, :-1]
        rew = rew[:, :-1]
        rtg = rtg[:, :-1]
        
        #################
        # forward
        # online encoder
        y_o, _ = self.model.backbone(x_o)  
        y_o = self.model.head.obs_to_latent(y_o)

        # decode
        dataset_type = self.cfg.dataset_type 
        dec_input = {'obs': y_o,
                     'act': act,
                     'rew': rew,
                     'rtg': rtg}

        dec_output = self.model.head.decode(dec_input, dataset_type) 
        obs_o, act_o, rew_o, rtg_o = dec_output['obs'], dec_output['act'], dec_output['rew'], dec_output['rtg']   
        if self.cfg.projection:
            z_o = self.model.head.project(obs_o)
        else:
            z_o = obs_o
        p_o = self.model.head.predict(z_o)
        z1_o, z2_o = z_o.chunk(2)
        p1_o, p2_o = p_o.chunk(2)
        
        # target encoder
        with torch.no_grad():
            y_t, _ = self.target_model.backbone(x_t)
            y_t = self.model.head.obs_to_latent(y_t)
            if self.cfg.projection:
                z_t = self.target_model.head.project(y_t)
            else:
                z_t = y_t
            z1_t, z2_t = z_t.chunk(2)

        #################
        # loss
        # observation loss (self-predictive)
        if self.cfg.obs_loss_type == 'consistency':
            obs_loss_fn = ConsistencyLoss()
        
        elif self.cfg.obs_loss_type == 'contrastive':
            obs_loss_fn = CURLLoss(self.cfg.temperature)
        
        p1_o, p2_o = rearrange(p1_o, 'n t d -> (n t) d'), rearrange(p2_o, 'n t d -> (n t) d')
        z1_t, z2_t = rearrange(z1_t, 'n t d -> (n t) d'), rearrange(z2_t, 'n t d -> (n t) d')

        obs_loss = 0.5 * (obs_loss_fn(p1_o, z2_t) + obs_loss_fn(p2_o, z1_t))
        obs_loss = torch.mean(obs_loss)

        # action loss
        if act_o is not None:
            act_loss_fn = nn.CrossEntropyLoss()
            act_o = rearrange(act_o, 'n t d -> (n t) d')
            act_t = rearrange(act, 'n t -> (n t)')
            act_loss = act_loss_fn(act_o, act_t)
            act_acc = torch.mean((torch.argmax(act_o, 1) == act_t).float())
        else:
            act_loss = torch.Tensor([0.0]).to(x.device)
            act_acc = torch.Tensor([0.0]).to(x.device)
            
        # idm loss
        idm_loss_fn = nn.CrossEntropyLoss()
        idm_o = self.model.head.idm_predictor(torch.cat((z_o[:, :-1, :], z_t[:, 1:, :]), dim=-1))
        idm_t = act[:, 1:]
        idm_o = rearrange(idm_o, 'n t d -> (n t) d')
        idm_t = rearrange(idm_t, 'n t -> (n t)')
        idm_loss = idm_loss_fn(idm_o, idm_t)
        idm_acc = torch.mean((torch.argmax(idm_o, 1) == idm_t).float())
        
        # reward loss
        if rew_o is not None:
            rew_loss_fn = nn.MSELoss()
            rew_o = rearrange(rew_o, 'n t 1 -> (n t 1)')
            rew = rearrange(rew, 'n t -> (n t)')
            rew_loss = rew_loss_fn(rew_o, rew)
        else:
            rew_loss = torch.Tensor([0.0]).to(x.device)
            
        # rtg loss
        if rtg_o is not None:
            rtg_loss_fn = nn.MSELoss()
            rtg_o = rearrange(rtg_o, 'n t 1 -> (n t 1)')
            rtg = rearrange(rtg, 'n t -> (n t)')
            rtg_loss = rtg_loss_fn(rtg_o, rtg)
        else:
            rtg_loss = torch.Tensor([0.0]).to(x.device)

        loss = (self.cfg.obs_lmbda * obs_loss + self.cfg.act_lmbda * act_loss + 
                self.cfg.idm_lmbda * idm_loss + self.cfg.rew_lmbda * rew_loss +
                self.cfg.rtg_lmbda * rtg_loss)
        
        ###############
        # logs
        # quantitative
        pos_idx = torch.eye(p1_o.shape[0], device=p1_o.device)
        sim = F.cosine_similarity(p1_o.unsqueeze(1), z2_t.unsqueeze(0), dim=-1)
        pos_sim = (torch.sum(sim * pos_idx) / torch.sum(pos_idx))
        neg_sim = (torch.sum(sim * (1-pos_idx)) / torch.sum(1-pos_idx))
        pos_neg_diff = pos_sim - neg_sim

        # qualitative
        # (n, t, f, c, h, w) -> (t, c, h, w)
        masked_frames = x_o[0, :, 0, :, :, :]
        target_frames = x_t[0, :, 0, :, :, :]
        
        log_data = {'loss': loss.item(),
                    'obs_loss': obs_loss.item(),
                    'act_loss': act_loss.item(),
                    'idm_loss': idm_loss.item(),
                    'rew_loss': rew_loss.item(),
                    'rtg_loss': rtg_loss.item(),
                    'act_acc': act_acc.item(),
                    'idm_acc': idm_acc.item(),
                    'pos_sim': pos_sim.item(),
                    'neg_sim': neg_sim.item(),
                    'pos_neg_diff': pos_neg_diff.item()}
                    #'masked_frames': wandb.Image(masked_frames),
                    #'target_frames': wandb.Image(target_frames)}
        
        return loss, log_data

    
    def update(self, obs, act, rew, done, rtg):
        tau = self.tau_scheduler.get_value()
        for online, target in zip(self.model.parameters(), self.target_model.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data

            
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

            
    """
    def evaluate_policy(self):
        self.model.eval()
        
        for _ in tqdm.tqdm(range(self.cfg.num_eval_trajectories)):
            # initialize history
            obs_list, act_list, rew_list, rtg_list = [], [], [], []

            # initialize tree
            rollout_cfg = self.cfg.rollout
            rollout_type = rollout_cfg.pop('type')
            
            if rollout_type == 'mcts':        
                raise NotImplemented
                    
            elif rollout_type == 'beam':
                tree = Beam(model=self.model, 
                            device=self.device,
                            action_size=self.cfg.action_size,
                            rtg_scale=self.max_rtg,
                            **rollout_cfg)
            
            # run trajectory
            obs = self.env.reset()
            act_sequence = []
            rtg = self.max_rtg
            while True:
                # encode obs
                obs = np.array(obs).astype(np.float32)
                obs = obs / 255.0
                obs = torch.FloatTensor(obs).to(self.device)
                obs = rearrange(obs, 'f c h w -> 1 1 f c h w')
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
                act_sequence = beam['act_batch'].numpy().tolist()
                
                # eps-greedy
                eps = self.cfg.eval_eps
                prob = random.random()
                argmax_action = act_sequence.pop(0)
                
                if prob < eps:
                    action = random.randint(0, self.cfg.action_size-1)
                else:
                    action = argmax_action

                # step
                next_obs, reward, done, info = self.env.step(action)
                
                # logger
                self.agent_logger.step(obs, reward, done, info)
                
                # move on
                if info.traj_done:
                    break
                else:
                    obs = next_obs
                    act_list.append(action)
                    rew_list.append(reward)
                    rtg_list.append(rtg / self.max_rtg)
                    rtg = rtg - reward
                
                #print(rtg)
                print(np.sum(self.agent_logger.traj_game_scores))

        log_data = self.agent_logger.fetch_log()
        
        return log_data
        """