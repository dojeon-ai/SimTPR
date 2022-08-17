from .base import BaseTrainer
from src.common.train_utils import LinearScheduler, CosineAnnealingWarmupRestarts
from src.common.train_utils import get_random_1d_mask, get_random_3d_mask
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange, repeat
from fractions import Fraction
import random
import copy
import tqdm


class MAETrainer(BaseTrainer):
    name = 'mae'
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
        self.optimizer = self._build_optimizer(cfg.optimizer)
        self.lr_scheduler = self._build_scheduler(self.optimizer, cfg.num_epochs, cfg.scheduler)
        
        self.cfg.patch_size = self.model.backbone.patch_size
        self.cfg.num_patches = self.model.backbone.num_patches
        
        assert cfg.pretrain_type in {'naive', 'freeze'}
        if cfg.pretrain_type == 'freeze':
            for param in self.model.backbone.encoder.parameters():
                param.requires_grad = False
            print('hi')
        
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
    
    def get_model_input(self, obs, act, done):
        # reshape obs for augmentation
        obs = rearrange(obs, 'n t s c h w -> (n t) (s c) h w')
        obs = obs.float() / 255.0
        
        # perform augmentation if needed
        aug_obs = self.aug_func(obs)
        aug_obs = rearrange(aug_obs, '(n t) c h w -> n t c h w', n=self.cfg.batch_size, t=self.cfg.t_step) 
        patch = rearrange(aug_obs, 'n t c (h p1) (w p2) -> n (t h w) (p1 p2 c)', 
                          p1 = self.cfg.patch_size[0], p2 = self.cfg.patch_size[1])        
        
        # construct input data for vit
        x = {
            'patch': patch,
            'act': act,
            'done': done,
        }
        return x
    
    def get_patch_mask(self):
        # construct mask data for vit-encoder
        video_shape = (self.cfg.batch_size, self.cfg.t_step, self.cfg.num_patches) 
        patch_ids_keep, patch_mask, patch_ids_restore = get_random_3d_mask(video_shape, self.cfg.patch_mask_ratio, self.cfg.patch_mask_type)
        
        # to device
        patch_ids_keep = patch_ids_keep.to(self.device)
        patch_mask = patch_mask.to(self.device)
        patch_ids_restore = patch_ids_restore.to(self.device)
        
        patch_mask_dict = {
            'patch_mask_type': self.cfg.patch_mask_type,
            'patch_mask': patch_mask,
            'patch_ids_keep': patch_ids_keep,
            'patch_ids_restore': patch_ids_restore,
        }
        return patch_mask_dict
    
    def get_act_mask(self):
        # construct mask data for vit-encoder
        act_shape = (self.cfg.batch_size, self.cfg.t_step-1) 
        if type(self.cfg.act_mask_ratio) == str:
            act_mask_ratio = float(Fraction(self.cfg.act_mask_ratio))
        
        act_ids_keep, act_mask, act_ids_restore = get_random_1d_mask(act_shape, act_mask_ratio)
        
        # to device
        act_ids_keep = act_ids_keep.to(self.device)
        act_mask = act_mask.to(self.device)
        act_ids_restore = act_ids_restore.to(self.device)
        
        act_mask_dict = {
            'act_mask': act_mask,
            'act_ids_keep': act_ids_keep,
            'act_ids_restore': act_ids_restore,
        }
        return act_mask_dict

    def compute_loss(self, obs, act, done):        
        # perform augmentation if needed
        x = rearrange(obs, 'n t c h w -> n (t c) h w')
        with torch.no_grad():
            x = self.aug_func(x)
        x = rearrange(x, 'n (t c) h w -> n t c h w', t=self.cfg.t_step) 
                
        # done-mask
        done_mask = torch.zeros((done.shape[0], self.cfg.t_step), device=done.device)
        done = done.float()
        done_idx = torch.nonzero(done==1)

        # done is masked in reverse-order is required to keep consistency with evaluation stage.
        for idx in done_idx:
            row = idx[0]
            col = idx[1]
            done_mask[row, :col+1] = 1
        
        # mask-out an input by masking
        x = x * (1-rearrange(done_mask, 'n t -> n t 1 1 1'))        
        
        ##################
        # patch loss
        if self.cfg.loss_type == 'patch':
            # get target
            patch = rearrange(obs, 'n t c (h p1) (w p2) -> n (t h w) (p1 p2 c)', 
                          p1 = self.cfg.patch_size[0], p2 = self.cfg.patch_size[1])
            
            # get patch mask
            patch_mask_dict = self.get_patch_mask()
            
            # predict masked patches
            x = self.model.backbone(x, patch_mask_dict=patch_mask_dict)
            patch_pred, _ = self.model.head(x, act, done, 
                                         use_action=False, patch_mask_dict=patch_mask_dict)

            # repeat for patches
            patch_mask = torch.repeat_interleave(done_mask, repeats=self.cfg.num_patches, dim=1)

            # do not reconstruct patches when done_mask=True
            patch_mask = patch_mask_dict['patch_mask'] * (1-patch_mask)
            
            # loss
            patch_loss = (patch_pred - patch) ** 2
            patch_loss = patch_loss.mean(dim=-1)

            # mean over removed patches
            loss = (patch_loss * patch_mask).sum() / patch_mask.sum()
        
            # logs
            log_data = {}
            log_data['patch_loss'] = loss.item()
        
        ##################
        # act loss
        elif self.cfg.loss_type == 'act':
            # get act mask
            act_mask_dict = self.get_act_mask()
            
            # predict masked patches
            x = self.model.backbone(x)
            _, act_pred = self.model.head(x, act, done, 
                                         use_action=True, act_mask_dict=act_mask_dict)
            
            # do not reconstruct actions when done_mask=True
            act_mask = act_mask_dict['act_mask'] * (1-done_mask[:, :-1])
            
            # loss
            act_pred = rearrange(act_pred, 'n t d -> (n t) d')
            act = rearrange(act, 'n t -> (n t)')
            act_mask = rearrange(act_mask, 'n t -> (n t)')
            act_loss = F.cross_entropy(act_pred, act, reduction='none')
            
            # mean over removed actions
            loss = (act_loss * act_mask).sum() / (act_mask.sum())
            
            # logs
            log_data = {}
            log_data['act_loss'] = loss.item()

            act_correct = torch.max(act_pred, 1)[1] == act
            log_data['act_acc'] = ((act_correct * act_mask).sum() / act_mask.sum()).item()
            
        return loss, log_data
        

    def train(self):
        self.model.train()
        loss, t = 0, 0
        for e in range(1, self.cfg.num_epochs+1):
            for batch in tqdm.tqdm(self.dataloader):                         
                # forward
                obs = batch.observation.to(self.device)
                obs = rearrange(obs, 'n t s c h w -> n t (s c) h w') / 255.0
                act = batch.action.to(self.device)[:, :-1]
                done = batch.done.to(self.device)[:, :-1]
                loss, log_data = self.compute_loss(obs, act, done)
                
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
                self.optimizer.step()

                # evaluation
                if t % self.cfg.eval_every == 0:
                    self.evaluate(obs, act, done)
                    self.model.train()
                    
                # log
                self.logger.update_log(**log_data)
                if t % self.cfg.log_every == 0:
                    self.logger.write_log()

                # proceed
                t += 1
                        
            if e % self.cfg.save_every == 0:
                self.logger.save_state_dict(model=self.model, epoch=e)

            self.lr_scheduler.step()

            
    def evaluate(self, obs, act, done):
        self.model.eval()
        with torch.no_grad():
            patch_mask_dict = self.get_patch_mask()  
            x = self.model.backbone(obs, patch_mask_dict=patch_mask_dict)
            patch_pred, _ = self.model.head(x, act, done.float(), 
                                         use_action=False, patch_mask_dict=patch_mask_dict)
            patch = rearrange(obs, 'n t c (h p1) (w p2) -> n (t h w) (p1 p2 c)', 
                          p1 = self.cfg.patch_size[0], p2 = self.cfg.patch_size[1])
        
        def depatchify(patch):
            video = rearrange(patch, 'n (t h w) (p1 p2 c) -> n t c (h p1) (w p2)', 
                              t=self.cfg.t_step, 
                              c = self.cfg.obs_shape[1], 
                              h=self.cfg.obs_shape[2]//self.cfg.patch_size[0], 
                              w=self.cfg.obs_shape[2]//self.cfg.patch_size[1], 
                              p1 = self.cfg.patch_size[0], 
                              p2 = self.cfg.patch_size[1])
            return video
        
        patch_mask = patch_mask_dict['patch_mask']
        
        target_video = (depatchify(patch)[0])
        masked_video = (depatchify(patch * (1-patch_mask).unsqueeze(-1))[0])
        
        patch_pred = patch_pred * patch_mask.unsqueeze(-1) + patch * (1-patch_mask).unsqueeze(-1)
        pred_video = (depatchify(patch_pred)[0]).to(float)
        pred_video = torch.where(pred_video>=1.0, 1.0, pred_video)
        pred_video = torch.where(pred_video<0.0, 0.0, pred_video)

        wandb.log({'target_video': wandb.Image(target_video),
                   'masked_video': wandb.Image(masked_video),
                   'pred_video': wandb.Image(pred_video)
                  }, step=self.logger.step)