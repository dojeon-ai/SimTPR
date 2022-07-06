import argparse
import hydra
from hydra import compose, initialize
from src.dataloaders import *
from src.envs import *
from src.models import *
from src.common.logger import WandbTrainerLogger
from src.common.utils import set_global_seeds
from src.agents import build_agent
from typing import List
from dotmap import DotMap
import torch
import numpy as np


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
    dataloader = build_dataloader(cfg.dataloader)

    for batch in dataloader:
        import pdb
        pdb.set_trace()


    # environment
    train_env, eval_env = build_env(cfg.env)
    cfg.agent.obs_shape = cfg.model.backbone.obs_shape = train_env.observation_space.shape
    cfg.agent.action_size = cfg.model.policy.action_size = train_env.action_space.n
    
    # logger
    logger= WandbTrainerLogger(cfg)

    # model
    model = build_model(cfg.model)

    if logger.use_pretrained_model:
        pretrained_model_path = logger.get_pretrained_model_path()
        checkpoint = logger.load_state_dict(pretrained_model_path)
        model.load_backbone_and_policy(checkpoint)

    # agent
    agent = build_agent(cfg=cfg.agent,
                        device=device,
                        train_env=train_env,
                        eval_env=eval_env,
                        logger=logger,
                        model=model)

    # train
    agent.train()
    return logger
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_dir',  type=str,    default='atari')
    parser.add_argument('--config_name', type=str,    default='mixed_byol_nature') 
    parser.add_argument('--overrides',   action='append', default=[])
    args = parser.parse_args()

    run(vars(args))