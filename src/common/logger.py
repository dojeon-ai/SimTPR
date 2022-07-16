import wandb
import torch
import os
import json
from omegaconf import OmegaConf
from src.common.utils import save__init__args
from collections import deque
import numpy as np


class WandbTrainerLogger(object):
    def __init__(self, cfg):
        self.cfg = cfg
        dict_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(project=cfg.project_name, 
                   config=dict_cfg,
                   group=cfg.exp_name,
                   settings=wandb.Settings(start_method="thread"))    

        self.model_path = wandb.run.dir + '/model.pth'
        self.config_path = wandb.run.dir + '/config.json'
        with open(self.config_path, 'w') as f:
            json.dump(dict_cfg, f)
        self.artifacts = {}

        self.logger = TrainerLogger()
        self.step = 0

    def update_log(self, **kwargs):
        self.logger.update_log(**kwargs)
        self.step += 1
    
    def write_log(self):
        log_data = self.logger.fetch_log()
        wandb.log(log_data, step=self.step)

    def save_state_dict(self, model, epoch=1):
        name = self.cfg.dataloader.game + '/' + str(self.cfg.seed) + '/' + str(epoch) + '/model.pth'
        path = wandb.run.dir + '/' + name
        _dir = os.path.dirname(path)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        state_dict = {'model_state_dict': model.state_dict()}
        torch.save(state_dict, path)
        self.artifacts[path] = name
    
    def load_state_dict(self, path):
        return torch.load(path)

    def get_artifacts(self):
        return self.artifacts


class TrainerLogger(object):
    def __init__(self):
        self.average_meter_set = AverageMeterSet()
    
    def update_log(self, **kwargs):
        for k, v in kwargs.items():
            self.average_meter_set.update(k, v)

    def fetch_log(self):
        log_data = {}
        log_data.update(self.average_meter_set.averages())
        self.average_meter_set = AverageMeterSet()
            
        return log_data


class WandbAgentLogger(object):
    def __init__(self, cfg):
        self.cfg = cfg
        dict_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(project=cfg.project_name, 
                   config=dict_cfg,
                   group=cfg.exp_name,
                   reinit=True,
                   settings=wandb.Settings(start_method="thread"))    

        self._use_pretrained_model = False
        if cfg.use_artifact:
            artifact = wandb.run.use_artifact(str(cfg.artifact_name) + ':latest')
            model_path = artifact.get_path(cfg.model_path).download()
            self.pretrained_model_path = model_path
            self._use_pretrained_model = True

        self.model_path = wandb.run.dir + '/model.pth'
        self.config_path = wandb.run.dir + '/config.json'
        with open(self.config_path, 'w') as f:
            json.dump(dict_cfg, f)
        self.artifacts = {}

        self.train_logger = AgentLogger(average_len=10)
        self.eval_logger = AgentLogger(average_len=100)
        self.timestep = 0
    
    def step(self, state, reward, done, info, mode='train'):
        if mode == 'train':
            self.train_logger.step(state, reward, done, info)
            self.timestep += 1

        elif mode == 'eval':
            self.eval_logger.step(state, reward, done, info)

    def update_log(self, mode='train', **kwargs):
        if mode == 'train':
            self.train_logger.update_log(**kwargs)

        elif mode == 'eval':
            self.eval_logger.update_log(**kwargs)
    
    def write_log(self, mode='train'):
        if mode == 'train':
            log_data = self.train_logger.fetch_log()

        elif mode == 'eval':
            log_data = self.eval_logger.fetch_log()

        # prefix
        log_data = {mode+'_'+k: v for k, v in log_data.items() }
        wandb.log(log_data, step=self.timestep)

    def save_state_dict(self, model):
        name = self.cfg.env.game + '/' + str(self.cfg.seed) + '/' + str(self.timestep) + '/model.pth'
        path = wandb.run.dir + '/' + name
        _dir = os.path.dirname(path)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        state_dict = {'model_state_dict': model.state_dict()}
        torch.save(state_dict, path)
        self.artifacts[path] = name
    
    def load_state_dict(self, path):
        return torch.load(path)

    def get_artifacts(self):
        return self.artifacts

    def get_mean_eval_score(self):
        return np.mean(self.eval_logger.traj_game_scores_buffer)

    def get_game_name(self):
        return self.cfg.env.game

    def get_pretrained_model_path(self):
        return self.pretrained_model_path

    @property
    def use_pretrained_model(self):
        return self._use_pretrained_model


class AgentLogger(object):
    def __init__(self, average_len=10):
        # https://arxiv.org/pdf/1709.06009.pdf 
        # Section 3.1 -> Training: end-of-life / Evaluation: end-of-trajectory
        # episode = life / traj = all lives
        self.traj_rewards = []
        self.traj_game_scores = []
        self.traj_rewards_buffer = deque(maxlen=average_len)
        self.traj_game_scores_buffer = deque(maxlen=average_len)
        self.average_meter_set = AverageMeterSet()
        
    def step(self, state, reward, done, info):
        self.traj_rewards.append(reward)
        self.traj_game_scores.append(info.game_score)

        if info.traj_done:
            self.traj_rewards_buffer.append(np.sum(self.traj_rewards))
            self.traj_game_scores_buffer.append(np.sum(self.traj_game_scores))
            self.traj_rewards = []
            self.traj_game_scores = []
    
    def update_log(self, **kwargs):
        for k, v in kwargs.items():
            self.average_meter_set.update(k, v)

    def fetch_log(self):
        log_data = {}
        log_data['mean_traj_rewards'] = np.mean(self.traj_rewards_buffer)
        log_data['mean_traj_game_scores'] = np.mean(self.traj_game_scores_buffer)
        log_data.update(self.average_meter_set.averages())
        self.average_meter_set = AverageMeterSet()
            
        return log_data


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # TODO: description for using n
        self.val = val
        self.sum += (val * n)
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)