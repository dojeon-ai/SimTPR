import subprocess
import argparse
import json
import copy
import wandb
import itertools
import os
import multiprocessing as mp
import multiprocessing
from src.envs.atari import *
from run_finetune import run
import numpy as np
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--group_name',   type=str,     default='test')
    parser.add_argument('--exp_name',     type=str,     default='test')
    parser.add_argument('--config_dir',   type=str,     default='atari/finetune')
    parser.add_argument('--config_name',  type=str,     default='simtpr')
    parser.add_argument('--games',        type=str,     default='[]') 
    parser.add_argument('--seeds',        type=str,     default='[1,2,3,4,5]')
    parser.add_argument('--num_devices',  type=int,     default=6)
    parser.add_argument('--num_exp_per_device',  type=int,  default=3)
    parser.add_argument('--overrides',    type=str,     default=[],      nargs='*') 

    args = vars(parser.parse_args())
    all_games = list(atari_human_scores.keys())
    
    seeds = eval(args.pop('seeds'))
    if type(seeds) != list:
        raise ValueError
    games = eval(args.pop('games'))
    if len(games) == 0:
        games = all_games

    num_devices = args.pop('num_devices')
    num_exp_per_device = args.pop('num_exp_per_device')
    pool_size = num_devices * num_exp_per_device 

    # create configurations for child run
    experiments = []
    for seed, game in itertools.product(*[seeds, games]):
        exp = copy.deepcopy(args)
        group_name = exp.pop('group_name')
        exp_name = exp.pop('exp_name')
        exp['overrides'].append('group_name=' + group_name)
        exp['overrides'].append('exp_name=' + exp_name)
        exp['overrides'].append('seed=' + str(seed))
        exp['overrides'].append('env.game=' + str(game))

        experiments.append(exp)
        print(exp)

    # run parallell experiments
    # https://docs.python.org/3.5/library/multiprocessing.html#contexts-and-start-methods
    mp.set_start_method('spawn') 
    available_gpus = list(range(num_devices))
    process_dict = {gpu_id: [] for gpu_id in available_gpus}

    for exp in experiments:
        wait = True
        # wait until there exists a finished process
        while wait:
            # Find all finished processes and register available GPU
            for gpu_id, processes in process_dict.items():
                for process in processes:
                    if not process.is_alive():
                        print(f"Process {process.pid} on GPU {gpu_id} finished.")
                        processes.remove(process)
                        if gpu_id not in available_gpus:
                            available_gpus.append(gpu_id)
            
            for gpu_id, processes in process_dict.items():
                if len(processes) < num_exp_per_device:
                    wait = False
                    gpu_id, processes = min(process_dict.items(), key=lambda x: len(x[1]))
                    break
            
            time.sleep(10)

        # get running processes in the gpu
        processes = process_dict[gpu_id]
        exp['overrides'].append('device=' + 'cuda:' + str(gpu_id))
        process = multiprocessing.Process(target=run, args=(exp, ))
        process.start()
        processes.append(process)
        print(f"Process {process.pid} on GPU {gpu_id} started.")

        # check if the GPU has reached its maximum number of processes
        if len(processes) == num_exp_per_device:
            available_gpus.remove(gpu_id)
