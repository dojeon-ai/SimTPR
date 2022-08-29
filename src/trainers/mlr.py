from .base import BaseTrainer
from src.common.train_utils import LinearScheduler, CosineAnnealingWarmupRestarts
from src.common.train_utils import get_random_1d_mask, get_random_3d_mask, get_3d_masked_input
from src.common.losses import TemporalConsistencyLoss, TemporalSimilarityLoss
from fractions import Fraction
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange, repeat
import random
import copy
import tqdm


class MLRTrainer(BaseTrainer):
    name = 'mlr'
    def __init__(self,
                 cfg,
                 device,
                 dataloader,
                 logger, 
                 aug_func,
                 model):
        
        super().__init__()  
        self.cfg = cfg  
        self.device = device
        self.dataloader = dataloader
        self.logger = logger

        self.aug_func = aug_func.to(self.device)
        self.model = model.to(self.device)
        self.target_model = copy.deepcopy(self.model).to(self.device)

        self.optimizer = self._build_optimizer(cfg.optimizer)
        self.lr_scheduler = self._build_scheduler(self.optimizer, cfg.num_epochs, cfg.scheduler)

        self.cfg.patch_size = self.model.head.patch_size

        

    def _build_optimizer(self, optimizer_cfg):
        optimizer_type = optimizer_cfg.pop('type')
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), 
                              **optimizer_cfg)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), 
                              **optimizer_cfg)
        else:
            raise ValueError

    def _build_scheduler(self, optimizer, num_epochs, scheduler_cfg):
        return CosineAnnealingWarmupRestarts(optimizer=optimizer,
                                             first_cycle_steps=num_epochs,
                                             **scheduler_cfg)
    
    def _update_moving_average(self):
        for online, target in zip(self.model.parameters(), self.target_model.parameters()):
            target.data = self.cfg.tau * target.data + (1 - self.cfg.tau) * online.data
    
    def forward_online_model(self, obs, act, done):

        # reshape aug_obs to patches
        patch_height, patch_width = self.cfg.patch_size
        batch, t_step, channel, image_height, image_width = obs.shape

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        patch = rearrange(obs, 'n t c (h p1) (w p2) -> n (t h w) (p1 p2 c)', 
                          p1 = patch_height, p2 = patch_width)
        
        # construct mask data for vit-encoder
        video_shape = (batch, t_step, num_patches)

        patch_ids_keep, patch_mask, patch_ids_restore = get_random_3d_mask(video_shape, self.cfg.patch_mask_ratio, self.cfg.patch_mask_type)
        
        # to device
        patch_ids_keep = patch_ids_keep.to(self.device)
        patch_mask = patch_mask.to(self.device)
        patch_ids_restore = patch_ids_restore.to(self.device)
        
        mask = {
            'patch_mask_type': self.cfg.patch_mask_type,
            'patch_ids_keep': patch_ids_keep,
            'patch_ids_restore': patch_ids_restore,
        }
        
        # masking and restore into images
        patch = rearrange(patch, 'n (t p) d -> n t p d', t = t_step, p = num_patches)
        original_size = (patch.shape[0], patch.shape[1] * patch.shape[2], patch.shape[3])
        patch = get_3d_masked_input(patch, mask['patch_ids_keep'], mask['patch_mask_type'])
        zero_padding_size = (original_size[0], original_size[1] - patch.shape[1], original_size[2])
        mask_tokens = torch.zeros(zero_padding_size).to(patch.device)
        patch = torch.cat([patch, mask_tokens], dim=1)
        patch = torch.gather(patch, dim=1, index=mask['patch_ids_restore'].unsqueeze(-1).repeat(1,1,patch.shape[-1]).to(patch.device))


        # augmentation
        x = rearrange(patch, 'n (t p1 p2) (d1 d2 c) -> n (t c) (p1 d1) (p2 d2)', t = t_step, p1= image_height // patch_height, p2 = image_width // patch_width, \
                            c = channel, d1 = patch_height, d2 = patch_width)
        x = self.aug_func(x)

        x = rearrange(x, 'n (t c) h w -> n t c h w', t=t_step) 

        done_mask = torch.zeros((done.shape[0], t_step), device=done.device)
        done = done.float()
        done_idx = torch.nonzero(done==1)

        # done is masked in reverse-order is required to keep consistency with evaluation stage.
        for idx in done_idx:
            row = idx[0]
            col = idx[1]
            done_mask[row, :col+1] = 1
        
        x = x * (1-rearrange(done_mask, 'n t -> n t 1 1 1'))      

        # import pdb; pdb.set_trace()

        x = rearrange(x, 'n t c h w -> (n t) c h w') 

        x = self.model.backbone(x)

        x = rearrange(x, '(n t) d -> n t d', n=batch, t= t_step)

        x = {
            'patch': x,
            'act': act,
            'done': done,
        }

        x = self.model.head(x, True)

        return x


    def _compute_similarity(self, obs, act, done):
        # online network
        N, T, S, C, H, W = obs.shape
        obs = obs.reshape(N, T, S*C, H, W) 
        obs = obs.float() / 255.0
        p_o = self.forward_online_model(obs, act[:, :-1], done[:, :-1])

        # target network
        obs2 = rearrange(obs, 'n t c h w -> (n t) c h w')
        aug_tgt = self.aug_func(obs2)
        y_t = self.target_model.backbone(aug_tgt)
        y_t = self.target_model.head.projection(y_t)
        z_t = self.target_model.head.project(y_t)

        z = torch.cat((p_o, z_t), dim = 0)

        # loss
        sim_fn = TemporalSimilarityLoss(num_trajectory=N,
                                        t_step=T,  
                                        device=self.device)


        #import pdb; pdb.set_trace()

        positive_sim, negative_sim = sim_fn(z, done)

        

        return positive_sim, negative_sim


    def compute_loss(self, obs, act, done):
        # obs
        N, T, S, C, H, W = obs.shape
        obs = rearrange(obs, 'n t s c h w -> n t (s c) h w')
        obs = obs.float() / 255.0
        
        # online network
        p_o = self.forward_online_model(obs, act[:, :-1], done[:, :-1])
        
        # target network
        with torch.no_grad():
            obs = rearrange(obs, 'n t c h w -> (n t) c h w')
            aug_tgt = self.aug_func(obs)
            y_t = self.target_model.backbone(aug_tgt)
            y_t = self.target_model.head.projection(y_t)
            z_t = self.target_model.head.project(y_t)
            
        loss_fn = TemporalConsistencyLoss(num_trajectory=N, 
                                          t_step=T, 
                                          device=self.device)
        
        loss = 1 + loss_fn(p_o, z_t, done)
        
        return loss
        

    def train(self):
        self.model.train()
        self.target_model.train()
        loss, t = 0, 0
        for e in range(1, self.cfg.num_epochs+1):
            for batch in tqdm.tqdm(self.dataloader):         

                log_data = {}

                # forward
                obs = batch.observation.to(self.device)
                act = batch.action.to(self.device)
                done = batch.done.to(self.device)
                _loss = self.compute_loss(obs, act, done)
                loss += _loss
                
                # backward
                if (t % self.cfg.update_freq == 0) and (t>0):
                    loss /= self.cfg.update_freq
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self._update_moving_average()
                    log_data['loss'] = loss.item()
                    loss = 0.0

                if t % self.cfg.eval_every == 0:
                    with torch.no_grad():
                        positive_sim, negative_sim = self._compute_similarity(obs, act, done)
                    log_data['positive_sim'] = positive_sim.item()
                    log_data['negative_sim'] = negative_sim.item()
                    log_data['pos_neg_diff'] = positive_sim.item() - negative_sim.item()
                
                    
                # log
                self.logger.update_log(**log_data)
                if t % self.cfg.log_every == 0:
                    self.logger.write_log()

                # proceed
                t += 1
            
            if e % self.cfg.save_every == 0:
                self.logger.save_state_dict(model=self.model, epoch=e)

            self.lr_scheduler.step()
