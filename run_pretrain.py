import argparse
import hydra
from hydra import compose, initialize
from src.dataloaders import *
from src.envs import *
from src.models import *
from src.common.logger import WandbTrainerLogger
from src.common.utils import set_global_seeds
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
    initialize(version_base=None, config_path=config_path) 
    cfg = compose(config_name=config_name, overrides=overrides)
    
    # device
    device = torch.device(cfg.device)

    # dataset
    torch.set_num_threads(1)#<- when dataset on disk
    cfg.dataloader.device = cfg.device
    dataloader = build_dataloader(cfg.dataloader)

    # shape config
    env, _ = build_env(cfg.env)
    cfg.trainer.obs_shape = cfg.model.backbone.obs_shape = env.observation_space.shape
    cfg.trainer.action_size = env.action_space.n
    del env
    
    # logger
    logger= WandbTrainerLogger(cfg)

    # model
    model = build_model(cfg.model)

    # agent
    cfg.trainer.time_span = cfg.dataloader.t_step
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
    parser.add_argument('--config_dir',  type=str,    default='atari')
    parser.add_argument('--config_name', type=str,    default='mixed_byol_impala') 
    parser.add_argument('--overrides',   action='append', default=[])
    args = parser.parse_args()

    run(vars(args))