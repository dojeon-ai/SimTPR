import subprocess
import argparse
import json
import copy
import wandb
import itertools
from multiprocessing import Pool
import numpy as np

atari_human_scores = dict(
    alien=7127.7, amidar=1719.5, assault=742.0, asterix=8503.3,
    bank_heist=753.1, battle_zone=37187.5, boxing=12.1,
    breakout=30.5, chopper_command=7387.8, crazy_climber=35829.4,
    demon_attack=1971.0, freeway=29.6, frostbite=4334.7,
    gopher=2412.5, hero=30826.4, jamesbond=302.8, kangaroo=3035.0,
    krull=2665.5, kung_fu_master=22736.3, ms_pacman=6951.6, pong=14.6,
    private_eye=69571.3, qbert=13455.0, road_runner=7845.0,
    seaquest=42054.7, up_n_down=11693.2
)

atari_random_scores = dict(
    alien=227.8, amidar=5.8, assault=222.4,
    asterix=210.0, bank_heist=14.2, battle_zone=2360.0,
    boxing=0.1, breakout=1.7, chopper_command=811.0,
    crazy_climber=10780.5, demon_attack=152.1, freeway=0.0,
    frostbite=65.2, gopher=257.6, hero=1027.0, jamesbond=29.0,
    kangaroo=52.0, krull=1598.0, kung_fu_master=258.5,
    ms_pacman=307.3, pong=-20.7, private_eye=24.9,
    qbert=163.9, road_runner=11.5, seaquest=68.4, up_n_down=533.4
)

def run_experiment(experiment):
    cmd = ['python', 'run.py']
    for key, value in experiment.items():
        if key == '--overrides':
            for v in value:
                cmd.append(key)
                cmd.append(v)
        else:
            cmd.append(key)
            cmd.append(value)
    return subprocess.check_output(cmd)


if __name__ == '__main__':
    default = {'--config_dir' : 'drq',
               '--config_name': 'atari100k_rainbow_nature'}
    seeds = ['0', '1', '2', '3', '4']
    games = list(atari_human_scores.keys())

    num_devices = 5
    num_exp_per_device = 3
    pool_size = num_devices * num_exp_per_device

    experiments = []
    device = 0
    for seed, game in itertools.product(*[seeds, games]):
        exp = copy.deepcopy(default)
        device_id = int(device % num_devices)
        exp['--device'] = 'cuda:' + str(device_id)
        exp['--overrides'] = ['seed=' + str(seed)] 
        exp['--overrides'].append('env.game=' + str(game)) 
        exp['--overrides'].append('exp_name=' + 'drq_rainbow') 
        # exp['--overrides'].append('agent.num_timesteps='+str(2000))

        experiments.append(exp)
        device += 1

    pool = Pool(pool_size)
    results = pool.map(run_experiment, experiments, chunksize=1)
    pool.close()

    config = {'exp_name': 'benchmark',
              'name': default['--config_dir'] + '_' + default['--config_name']}

    wandb.init(project='atari100k', 
               config=config,
               settings=wandb.Settings(start_method="thread"))   

    logs = {}
    scores = {}
    for game in games:
        scores[game] = []

    for result in results:
        result = result.decode('utf-8')
        game, score = result.strip().split(':')
        scores[game].append(float(score))

    wandb_score_path = wandb.run.dir + '/scores.json'
    with open(wandb_score_path, 'w') as f:
        json.dump(scores, f)
    
    norm_scores = {}
    for game, score in scores.items():
        random_score = atari_random_scores[game] 
        human_score = atari_human_scores[game]

        normalized_score = (np.mean(score) - random_score) / (human_score - random_score)
        norm_scores[game] = normalized_score
        logs[game] = score
    
    _norm_scores = list(norm_scores.values())
    median = np.median(_norm_scores)
    mean = np.mean(_norm_scores)
    logs['Median-HNS'] = median
    logs['Mean-HNS'] = mean

    wandb.log(logs)
