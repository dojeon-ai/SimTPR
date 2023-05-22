import pandas as pd 
import pathlib
import wandb
import argparse
import shutil
import os
from dotmap import DotMap


def download(args):    
    args = DotMap(args)
    project_name = args.project_name
    group_name = args.group_name
    exp_name = args.exp_name
    model_path = args.model_path
    
    ##########################
    # collect runs from wandb
    api = wandb.Api()
    runs = api.runs(project_name)

    summary_list, config_list, id_list = [], [], []
    for run in runs: 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        id_list.append(run.id)

    data_ = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "id": id_list,
        })

    ##########################
    # filter runs
    configs = data_['config']
    indexs = []
    for idx, cfg in enumerate(configs):
        if len(cfg) == 0:
            continue
            
        run_exp_name = cfg['exp_name']
        run_group_name = cfg['group_name']

        # condition
        if run_exp_name == exp_name and run_group_name == group_name:
            if 'env' in cfg:
                indexs.append(idx)
    
    data = data_.iloc[indexs]
    envs = []
    for cfg in data['config']:
        envs.append(cfg['env']['game'])
    data['env'] = envs
    print('number of runs to download:', len(data))
    
    #######################
    # download
    base_path = str(pathlib.Path().resolve())

    def move_file(source, destination):
        # Create the destination directory if it doesn't exist
        os.makedirs(destination, exist_ok=True)
        
        # Move the file to the destination
        shutil.move(source, destination)

    for run_id, env in zip(data['id'], data['env']):
        print(env, run_id)
        src_path = env + '/' + model_path 
        dst_path = 'models/' + exp_name + '/' + src_path
        
        wandb.restore(src_path, run_path=project_name + '/' + run_id)            
        move_file(src_path, dst_path)
        shutil.rmtree(env)
                
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--project_name',  type=str,    default='simtpr_icml2023')
    parser.add_argument('--group_name',    type=str,    default='test') 
    parser.add_argument('--exp_name',      type=str,    default='simtpr')
    parser.add_argument('--model_path',    type=str,    default='0/10/model.pth')
    args = parser.parse_args()

    download(vars(args))
             