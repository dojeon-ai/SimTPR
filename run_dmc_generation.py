import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
#os.environ['MUJOCO_GL'] = 'egl'

import ipdb
from dotmap import DotMap
import torch
import argparse
import wandb
import datetime
from hydra import compose, initialize
from pathlib import Path

from src.envs import *
from src.models import *
from src.common.logger import WandbAgentLogger
from src.common.utils import set_global_seeds
from src.common.dmc_video import VideoRecorder
from src.agents import build_agent

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

    # seed TODO: needed?
    # set_global_seeds(cfg.seed)

    # environment
    cfg.agent.num_timesteps = cfg.env.num_timesteps
    cfg.agent.stddev_schedule = cfg.env.stddev_schedule
    cfg.agent.action_repeat = cfg.env.action_repeat
    cfg.env.seed = cfg.seed
    train_env, eval_env = build_env(cfg.env)
    # define obs_shape, action_shape
    cfg.agent.obs_shape = cfg.model.backbone.obs_shape = train_env.observation_space.shape
    cfg.agent.action_shape = cfg.model.policy.action_shape = train_env.action_space.shape
    cfg.agent.buffer.frame_stack = cfg.env.frame_stack
    cfg.agent.game = cfg.env.game
    # walker_run, quadruped_run -> differs in n_step and buffer_size with other envs # FIXME: hydra 자체 기능으로 처리할 수 있을것 같은데..
    if 'n_step' in cfg.env.keys():  # walker_run
        cfg.agent.buffer.n_step = cfg.env.n_step
    if 'batch_size' in cfg.env.keys():  # walker_run
        cfg.agent.batch_size = cfg.env.batch_size
    if 'buffer_size' in cfg.env.keys():  # walker_run
        cfg.agent.buffer.buffer_size = cfg.env.buffer_size

    # logger
    logger= WandbAgentLogger(cfg)
    work_dir = Path.cwd() / 'exps' / f'{cfg.env.game}' / datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    print('video save directory:', work_dir)
    video_dir = work_dir / f'seed_{cfg.seed}'
    video_recorder = VideoRecorder(save_dir=video_dir)

    # model
    model = build_model(cfg.model)

    # agent
    agent = build_agent(cfg=cfg.agent,
                        device=device,
                        train_env=train_env,
                        eval_env=eval_env,
                        logger=logger,
                        model=model,
                        video_recorder=video_recorder)


    # train
    agent.train()
    wandb.finish()
    return logger
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_dir',  type=str,    default='dmc/scratch')
    parser.add_argument('--config_name', type=str,    default='ddpg') 
    parser.add_argument('--overrides',   action='append', default=[])
    args = parser.parse_args()

    run(vars(args))


