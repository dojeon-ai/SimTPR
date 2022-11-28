import argparse
import hydra
from hydra import compose, initialize
from src.dataloaders import *
from src.envs import *
from src.envs.vec_env import VecEnv
from src.models import *
from src.common.logger import WandbTrainerLogger, AgentLogger, VecAgentLogger
from src.common.train_utils import set_global_seeds
from src.trainers import build_trainer
from typing import List
from dotmap import DotMap
import torch
import wandb
import numpy as np
import re


def run(args):    
    args = DotMap(args)
    config_dir = args.config_dir
    config_name = args.config_name
    overrides = args.overrides

    # Hydra Compose
    config_path = './configs/' + config_dir 
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path=config_path) 
    cfg = compose(config_name=config_name, overrides=overrides)
    
    # device
    device = torch.device(cfg.device)

    # dataset
    torch.set_num_threads(1) # when dataset on disk
    cfg.dataloader.device = cfg.device
    train_loader, eval_act_loader, eval_rew_loader = build_dataloader(cfg.dataloader)
    
    # shape config
    cfg.env.game = cfg.dataloader.game
    env, _ = build_env(cfg.env)
    obs_shape = [cfg.dataloader.train.frame] + list(env.observation_space.shape[1:])
    action_size = env.action_space.n
    
    # initiaize not pre-defined hyperparameters
    param_dict = {'obs_shape': obs_shape,
                  'action_size': action_size,
                  't_step': cfg.dataloader.train.t_step,
                  'batch_size': cfg.dataloader.train.batch_size}

    for key, value in param_dict.items():
        if key in cfg.model.backbone:
            cfg.model.backbone[key] = value
            
        if key in cfg.model.head:
            cfg.model.head[key] = value
            
        if key in cfg.model.policy:
            cfg.model.policy[key] = value
            
        if key in cfg.trainer:
            cfg.trainer[key] = value
    
    # logger
    logger= WandbTrainerLogger(cfg)
    agent_logger = AgentLogger(average_len=100)

    # model
    model = build_model(cfg.model)
    
    # load pretrained
    p_cfg = cfg.pretrain
    p_cfg.env = cfg.dataloader.game

    if p_cfg.use_pretrained:
        artifact = wandb.run.use_artifact(str(p_cfg.artifact_name))
        model_path = p_cfg.env + '/' + p_cfg.seed + '/' + p_cfg.name
        model_path = artifact.get_path(model_path).download()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'], strict=False)
        
    # trainer
    num_envs = cfg.trainer.num_envs
    if num_envs > 1:
        envs = []
        for idx in range(num_envs):
            cfg.env.seed = idx
            env, _ = build_env(cfg.env)
            envs.append(env)
        env = VecEnv(num_processes = num_envs, envs = envs)
        agent_logger = VecAgentLogger(average_len=100, num_envs= num_envs)
    
    trainer = build_trainer(cfg=cfg.trainer,
                            train_loader=train_loader,
                            eval_act_loader=eval_act_loader,
                            eval_rew_loader=eval_rew_loader,
                            env=env,
                            device=device,
                            logger=logger,
                            agent_logger=agent_logger,
                            model=model)
    
    # train
    if cfg.debug:
        trainer.debug()
    else:
        trainer.train()

    wandb.finish()
    return logger
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_dir',  type=str,    default='atari/pretrain')
    parser.add_argument('--config_name', type=str,    default='mixed_clt_impala') 
    parser.add_argument('--overrides',   action='append', default=[])
    args = parser.parse_args()

    run(vars(args))
             