from abc import *
from typing import Tuple
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.common.train_utils import CosineAnnealingWarmupRestarts, get_grad_norm_stats
from src.common.losses import SoftmaxFocalLoss
from sklearn.metrics import f1_score
from einops import rearrange
import cyanure as cy


class BaseTrainer():
    def __init__(self,
                 cfg,
                 device,
                 train_loader,
                 eval_act_loader,
                 eval_rew_loader,
                 logger,
                 aug_func,
                 model):
        super().__init__()
        self.cfg = cfg  
        self.device = device
        self.train_loader = train_loader
        self.eval_act_loader = eval_act_loader
        self.eval_rew_loader = eval_rew_loader
        self.logger = logger
        self.aug_func = aug_func.to(self.device)
        self.model = model.to(self.device)
        self.optimizer = self._build_optimizer(cfg.optimizer)
        self.lr_scheduler = self._build_scheduler(self.optimizer, cfg.scheduler)
        
    @classmethod
    def get_name(cls):
        return cls.name

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

    def _build_scheduler(self, optimizer, scheduler_cfg):
        first_cycle_steps = len(self.train_loader) * self.cfg.num_epochs
        return CosineAnnealingWarmupRestarts(optimizer=optimizer,
                                             first_cycle_steps=first_cycle_steps,
                                             **scheduler_cfg)
 
    @abstractmethod
    def compute_loss(self, obs, act, rew, done) -> Tuple[torch.Tensor, dict]:
        pass
    
    @abstractmethod
    # custom model update other than backpropagation (e.g., ema)
    def update(self, obs, act, rew, done):
        pass
    
    def debug(self):
        for batch in tqdm.tqdm(self.train_loader):   
            # forward
            self.model.train()
            obs = batch.observation.to(self.device)
            act = batch.action.to(self.device)
            rew = batch.reward.to(self.device)
            done = batch.done.to(self.device)
            loss, train_logs = self.compute_loss(obs, act, rew, done)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.update(obs, act, rew, done)

            # log         
            self.logger.update_log(**train_logs)
            break

        self.model.eval()
        eval_logs = self.evaluate()
        self.logger.update_log(**eval_logs)
        self.logger.write_log(step=0)
    
    def train(self):
        step = 0
        
        # initial evaluation
        self.model.eval()
        eval_logs = self.evaluate()
        best_metric_val = eval_logs[self.cfg.base_metric]
        eval_logs['epoch'] = 0
        eval_logs['best_metric_val'] = best_metric_val
        self.logger.update_log(**eval_logs)
        self.logger.write_log(step)
        
        # train
        for e in range(1, self.cfg.num_epochs+1):
            for batch in tqdm.tqdm(self.train_loader):   
                # forward
                self.model.train()
                obs = batch.observation.to(self.device)
                act = batch.action.to(self.device)
                rew = batch.reward.to(self.device)
                done = batch.done.to(self.device)
                loss, train_logs = self.compute_loss(obs, act, rew, done)
                
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                grad_stats = get_grad_norm_stats(self.model)
                train_logs.update(grad_stats)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
                self.optimizer.step()
                self.update(obs, act, rew, done)
                    
                # log         
                self.logger.update_log(**train_logs)
                if step % self.cfg.log_every == 0:
                    self.logger.write_log(step)
                    
                # proceed
                self.lr_scheduler.step()
                step += 1
            
            if e % self.cfg.eval_every == 0:
                self.model.eval()
                eval_logs = self.evaluate()
                eval_logs['epoch'] = e
                metric_val = eval_logs[self.cfg.base_metric]
                if metric_val > best_metric_val:
                    best_metric_val = metric_val
                    self.logger.save_state_dict(model=self.model, name='best')
                eval_logs['best_metric_val'] = best_metric_val
                    
                self.logger.update_log(**eval_logs)
                self.logger.write_log(step)
                
            if e % self.cfg.save_every == 0:
                self.logger.save_state_dict(model=self.model, name=e)
                
                
    def evaluate(self) -> dict:
        eval_logs = {}
        rew_eval_logs = self.probe_reward()
        act_eval_logs = self.probe_action()
        eval_logs.update(rew_eval_logs)
        eval_logs.update(act_eval_logs)

        return eval_logs
    
    ######################################################
    # pre-defined evaluation protocol from Zhang et al
    # https://arxiv.org/pdf/2208.12345.pdf
    def _generate_features_for_probing(self, task='reward'):
        xs, ys = [], []
        if task == 'reward':
            loader = self.eval_rew_loader
        elif task == 'action':
            loader = self.eval_act_loader
            
        for batch in tqdm.tqdm(loader):   
            obs = batch.observation.to(self.device)
            if task == 'reward':
                y = batch.reward[:, -1]       
            elif task == 'action':
                y = batch.action[:, -1]
            
            with torch.no_grad():
                x = obs / 255.0
                x, _ = self.model.backbone(x)
                x = rearrange(x, 'n t d -> n (t d)')
            
            xs.append(x.cpu().numpy())
            ys.append(y.cpu().numpy())
            
        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
        if task == 'reward':
            ys = (ys!=0).astype(np.float32)

        # train-test split
        test_size = 0.2
        split_idx = int(len(xs)*test_size)
        x_train = xs[split_idx:]
        y_train = ys[split_idx:]
        x_test = xs[:split_idx]
        y_test = ys[:split_idx]
    
        return x_train, y_train, x_test, y_test
    
    def probe_reward(self):
        print(f'start reward probing')
        x_train, y_train, x_test, y_test = self._generate_features_for_probing('reward')

        # logistic regression
        classifier = cy.estimators.Classifier(
            loss="logistic",
            penalty="l2",
            max_iter=300,
            tol=1e-5,
            verbose=False,
            fit_intercept=False,
            lambda_1=0.000000004,
        )

        cy.data_processing.preprocess(x_train, centering=True, normalize=True, columns=False)
        cy.data_processing.preprocess(x_test, centering=True, normalize=True, columns=False)
        
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        
        f1 = f1_score(y_test, y_pred)
        log_data = {}
        log_data['reward_ratio'] = np.sum(y_test) / len(y_test)
        log_data['reward_f1'] = f1

        return log_data
    
    def probe_action(self):
        print(f'start action probing')
        x_train, y_train, x_test, y_test = self._generate_features_for_probing('action')

        # Dataset & DataLoader
        class ActDataset(Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __getitem__(self, index):
                x = torch.from_numpy(self.x[index])
                y = torch.from_numpy(np.array(self.y[index])).long()
                return x, y

            def __len__(self):
                return len(self.x)
        
        train_dataset = ActDataset(x_train, y_train)
        test_dataset = ActDataset(x_test, y_test)
        train_dataloader = DataLoader(train_dataset, batch_size=256)
        test_dataloader = DataLoader(test_dataset, batch_size=256)

        # linear model & optimizer
        model = nn.Linear(x_train.shape[-1], self.cfg.action_size).to(self.device)
        criterion = SoftmaxFocalLoss(gamma=2)
        optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=1e-6)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
        
        # train
        epoch_ft = 50
        for epoch in range(epoch_ft):
            for x_batch, y_batch in train_dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                act_logits = model(x_batch)
                loss = criterion(act_logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
            lr_scheduler.step()
            
        # test
        y_pred = []
        for x_batch, y_batch in test_dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
                
            act_logits = model(x_batch)
            y_pred.append(torch.argmax(act_logits,1).cpu().numpy())
        
        y_pred = np.concatenate(y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')        
        log_data = {}
        log_data['act_f1'] = f1

        return log_data
