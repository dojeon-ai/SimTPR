from .base import BaseTrainer
from src.common.train_utils import LinearScheduler, CosineAnnealingWarmupRestarts
from src.common.train_utils import get_random_1d_mask, get_random_3d_mask, get_3d_masked_input
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
    
    def forward_model(self, obs, act, done):
        # reshape obs for augmentation
        obs = rearrange(obs, 'n t s c h w -> (n t) (s c) h w')
        obs = obs.float() / 255.0
        
        # perform augmentation if needed
        aug_obs = self.aug_func(obs)
        aug_obs = rearrange(aug_obs, '(n t) c h w -> n t c h w', n=self.cfg.batch_size, t=self.cfg.t_step) 

        patch_height, patch_width = self.cfg.patch_size
        batch, t_step, channel, image_height, image_width = aug_obs.shape

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        patch = rearrange(aug_obs, 'n t c (h p1) (w p2) -> n (t h w) (p1 p2 c)', 
                          p1 = patch_height, p2 = patch_width)
        
        # x = {
        #     'patch': patch,
        #     'act': act
        # }
        
        # construct mask data for vit-encoder
        

        video_shape = (batch, t_step, num_patches)

        patch_ids_keep, patch_mask, patch_ids_restore = get_random_3d_mask(video_shape, self.cfg.patch_mask_ratio, self.cfg.patch_mask_type)
        
        # # to device
        patch_ids_keep = patch_ids_keep.to(self.device)
        patch_mask = patch_mask.to(self.device)
        patch_ids_restore = patch_ids_restore.to(self.device)
        
        mask = {
            'patch_mask_type': self.cfg.patch_mask_type,
            'patch_ids_keep': patch_ids_keep,
            'patch_ids_restore': patch_ids_restore,
        }
        
        import pdb; pdb.set_trace()

        patch = rearrange(patch, 'n (t p) d -> n t p d', t = t_step, p = num_patches)
        original_size = (patch.shape[0], patch.shape[1] * patch.shape[2], patch.shape[3])
        patch = get_3d_masked_input(patch, mask['patch_ids_keep'], mask['patch_mask_type'])
        zero_padding_size = (original_size[0], original_size[1] - patch.shape[1], original_size[2])
        mask_tokens = torch.zeros(zero_padding_size).to(patch.device)
        patch = torch.cat([patch, mask_tokens], dim=1)
        patch = torch.gather(patch, dim=1, index=mask['patch_ids_restore'].unsqueeze(-1).repeat(1,1,patch.shape[-1]).to(patch.device))
        x = rearrange(patch, 'n (t p1 p2) (d1 d2 c) -> (n t) c (p1 d1) (p2 d2)', t = t_step, p1= image_height // patch_height, p2 = image_width // patch_width, \
                            c = channel, d1 = patch_height, d2 = patch_width)

        x = self.model.backbone(x)

        x = {
            'patch': x,
            'act': act,
            'done': done,
        }

        x = self.model.head(x)
        # construct input data for mlr

    
        # patch_pred, act_pred = self.model.backbone.predict(x)
        
        # return patch, patch_mask, patch_pred, act, act_mask, act_pred

    def compute_loss(self, obs, act, done):
        #patch, patch_mask, patch_pred, act, act_mask, act_pred = 
        self.forward_model(obs, act, done)
        
        # # loss
        # patch_loss = (patch_pred - patch) ** 2
        # patch_loss = patch_loss.mean(dim=-1)
        
        # act_pred = rearrange(act_pred, 'n t d -> (n t) d')
        # act = rearrange(act, 'n t -> (n t)')
        # act_mask = rearrange(act_mask, 'n t -> (n t)')
        # act_loss = F.cross_entropy(act_pred, act, reduction='none')
        
        # # mean over removed patches
        # #patch_loss = (patch_loss * patch_mask).sum() / patch_mask.sum()
        # patch_loss = patch_loss.mean()
        # act_loss = (act_loss * act_mask).sum() / act_mask.sum()
        
        # loss = patch_loss + act_loss
        
        # # log metrics
        # log_data = {}
        # log_data['loss'] = loss.item()
        # log_data['patch_loss'] = patch_loss.item()
        # log_data['act_loss'] = act_loss.item()
        
        # act_correct = torch.max(act_pred, 1)[1] == act
        # log_data['act_acc'] = (act_correct * act_mask).sum() / act_mask.sum()

        # return loss, log_data
        

    def train(self):
        self.model.train()
        loss, t = 0, 0
        for e in range(1, self.cfg.num_epochs+1):
            for batch in tqdm.tqdm(self.dataloader):                
                # forward
                obs = batch.observation.to(self.device)
                act = batch.action.to(self.device)
                done = batch.done.to(self.device)
                _loss, log_data = self.compute_loss(obs, act, done)
                loss += _loss
                
                # backward
                if (t % self.cfg.update_freq == 0) and (t>0):
                    loss /= self.cfg.update_freq
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
                    self.optimizer.step()
                    loss = 0.0

                # evaluation
                if t % self.cfg.eval_every == 0:
                    self.evaluate(obs, act, done)
                    
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
            patch, patch_mask, patch_pred, act, act_mask, act_pred = self.forward_model(obs, act, done)
        
        def depatchify(patch):
            video = rearrange(patch, 'n (t h w) (p1 p2 c) -> n t c (h p1) (w p2)', 
                              t=self.cfg.t_step, 
                              c = self.cfg.obs_shape[1], 
                              h=self.cfg.obs_shape[2]//self.cfg.patch_size[0], 
                              w=self.cfg.obs_shape[2]//self.cfg.patch_size[1], 
                              p1 = self.cfg.patch_size[0], 
                              p2 = self.cfg.patch_size[1])
            return video
        
        
        
        target_video = (depatchify(patch)[0])
        masked_video = (depatchify(patch * (1-patch_mask).unsqueeze(-1))[0])
        pred_video = (depatchify(patch_pred)[0]).to(float)
        pred_video = torch.where(pred_video>=1.0, 1.0, pred_video)
        pred_video = torch.where(pred_video<0.0, 0.0, pred_video)

        wandb.log({'target_video': wandb.Image(target_video),
                   'masked_video': wandb.Image(masked_video),
                   'pred_video': wandb.Image(pred_video)
                  }, step=self.logger.step)