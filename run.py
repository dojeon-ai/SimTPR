import argparse
from hydra import compose, initialize
from src.envs import *
from src.models import *
from src.common.logger import WandbLogger
from src.common.utils import set_global_seeds
from src.agents import build_agent
from typing import List
import torch
import numpy as np


def main(sys_argv: List[str] = None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_dir',  type=str,    default='rainbow')
    parser.add_argument('--config_name', type=str,    default='atari100k_der') 
    parser.add_argument('--device',      type=str,    default='cuda:0') 
    parser.add_argument('--overrides',   action='append', default=[])
    args = parser.parse_args()
    
    # Hydra Compose
    config_path = './configs/' + args.config_dir 
    initialize(version_base=None, config_path=config_path) 
    cfg = compose(config_name=args.config_name, overrides=args.overrides)
    
    # device
    device = torch.device(args.device)

    # environment
    train_env, eval_env = build_env(cfg.env)
    cfg.agent.obs_shape = cfg.model.backbone.obs_shape = train_env.observation_space.shape
    cfg.agent.action_size = cfg.model.policy.action_size = train_env.action_space.n
    
    # logger
    logger= WandbLogger(cfg)

    # model
    model = build_model(cfg.model)

    # agent
    agent = build_agent(cfg=cfg.agent,
                        device=device,
                        train_env=train_env,
                        eval_env=eval_env,
                        logger=logger,
                        model=model)

    # train
    agent.train()
    agent.evaluate()

    print(cfg.env.game + ':' + str(np.mean(agent.logger.eval_logger.traj_game_scores_buffer)))
    

if __name__ == '__main__':
    main()