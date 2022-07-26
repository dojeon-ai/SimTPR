import argparse
import hydra
from hydra import compose, initialize
from src.envs import *
from src.models import *
from src.common.logger import WandbAgentLogger
from src.common.utils import set_global_seeds
from src.agents import build_agent
from typing import List
from dotmap import DotMap
import torch
import wandb
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

    # environment
    train_env, eval_env = build_env(cfg.env)
    cfg.agent.obs_shape = cfg.model.backbone.obs_shape = train_env.observation_space.shape
    cfg.agent.action_size = cfg.model.policy.action_size = train_env.action_space.n
    
    # logger
    logger= WandbAgentLogger(cfg)

    # model
    model = build_model(cfg.model)

    # load-pretrained
    if logger.use_pretrained_model:
        pretrained_model_path = logger.get_pretrained_model_path()
        state_dict = logger.load_state_dict(pretrained_model_path)
        model.load_backbone(state_dict)

    # agent
    agent = build_agent(cfg=cfg.agent,
                        device=device,
                        train_env=train_env,
                        eval_env=eval_env,
                        logger=logger,
                        model=model)

    # train
    agent.train()
    wandb.finish()
    return logger
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_dir',  type=str,    default='atari/finetune')
    parser.add_argument('--config_name', type=str,    default='drq_impala') 
    parser.add_argument('--overrides',   action='append', default=[])
    args = parser.parse_args()

    run(vars(args))