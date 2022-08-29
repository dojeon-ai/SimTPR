import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import ipdb
from dotmap import DotMap
import torch
import argparse
import wandb
import datetime
from hydra import compose, initialize
from pathlib import Path
from typing import List

from src.envs import *
from src.models import *
from src.common.logger import WandbTrainerLogger
from src.common.utils import set_global_seeds
from src.common.dmc_video import VideoRecorder # TODO:
from src.common.logger import WandbTrainerLogger
from src.dataloaders import *
from src.trainers import build_trainer


def run(args):    
    args = DotMap(args)
    config_dir = args.config_dir
    config_name = args.config_name
    overrides = args.overrides

    # Hydra Compose
    config_path = './configs/' + config_dir 
    initialize(version_base=None, config_path=config_path) 
    cfg = compose(config_name=config_name, overrides=overrides)
    
    # device
    device = torch.device(cfg.device)

    # dataset
    torch.set_num_threads(1)#<- when dataset on disk
    cfg.dataloader.device = cfg.device
    cfg.dataloader.game = cfg.env.game

    dataloader = build_dataloader(cfg.dataloader)

    if 't_step' in cfg.model.backbone:
        cfg.trainer.t_step = cfg.model.backbone.t_step = cfg.dataloader.t_step
    else:
        cfg.trainer.t_step = cfg.dataloader.t_step

    # shape config
    cfg.env.seed = cfg.seed
    env, _ = build_env(cfg.env)
    obs_shape = [cfg.dataloader.frames] + list(env.observation_space.shape[1:])
    cfg.trainer.obs_shape = cfg.model.backbone.obs_shape = obs_shape  
    # TODO: action -> DMC는 dimension이 달라 (atari-> 1dim,categorical , DMC -> N-dim, continuous)
    cfg.trainer.action_size = cfg.model.backbone.action_size = env.action_space.shape
    ipdb.set_trace()
    del env
    
    # logger
    logger= WandbTrainerLogger(cfg)

    # model
    model = build_model(cfg.model)
    
    # agent   
    trainer = build_trainer(cfg=cfg.trainer,
                            dataloader=dataloader,
                            device=device,
                            logger=logger,
                            model=model)

    # train
    trainer.train()
    wandb.finish()
    return logger
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_dir',  type=str,    default='dmc/pretrain')
    parser.add_argument('--config_name', type=str,    default='mixed_simclr_cnn') 
    parser.add_argument('--overrides',   action='append', default=[])
    args = parser.parse_args()

    run(vars(args))