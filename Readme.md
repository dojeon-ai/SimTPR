```
project
|   README.md
|   run_pretrain.py
|   run_finetune.py
|   run_atari_pretrain.py  # pretrain 26 atari games w/ mp
|   run_atari_finetune.py  # finetune 26 atari games w/ mp
|---scripts
|   |  test.sh  # bash script for group of experiments
|   |  ...
|---configs
|   |---atari
|   |   |---pretrain
|   |   |   |  config.yaml
|   |   |   |---dataloader
|   |   |   |   |   mixed.yaml
|   |   |   |   |   expert.yaml
|   |   |   |---env
|   |   |   |   |   atari.yaml
|   |   |   |---model
|   |   |   |   nature.yaml
|   |   |   |   impala.yaml
|   |   |   |   ...
|   |   |   |---trainer
|   |   |   |   |   simclr.yaml
|   |   |   |   |   byol.yaml
|   |   |   |   |   ...
|   |   |---finetune
|   |   |   |  config.yaml
|   |   |   |---agent
|   |   |   |   |   drq.yaml
|   |   |   |   |   rainbow.yaml
|   |   |   |---env
|   |   |   |   |   atari.yaml
|   |   |   |---model
|   |   |   |   nature.yaml
|   |   |   |   impala.yaml
|   |   |   |   ...
|   |--- ...
|   
└───src
    |--- agents
    |--- envs
    |--- models
    |--- datasets
    |--- offline_learners
    |--- common
```
### Installation
#### ATARI
```
download_rom.ipynb
```
#### Mujoco
```
install mujoco
get mjkey.txt in .mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'/home/nas3_userK/hojoonlee/.mujoco/mujoco210/bin'
export MUJOCO_PY_MUJOCO_PATH='/home/nas3_userK/hojoonlee/.mujoco/mujoco210/'
export MUJOCO_PY_MJKEY_PATH='/home/nas3_userK/hojoonlee/.mujoco/mjkey.txt'
pip uninstall opencv-python
pip install opencv-python-headless
sudo apt-get install libglew-dev
sudo apt-get install -y patchelf
```

### Downloading Dataset
#### ATARI
```
pip install gsutil
export data_dir = 'custom_path'
bash download_atari_replay_data.sh
```

#### Mujoco
```
git clone https://github.com/rail-berkeley/d4rl.git
cd d4rl
pip install -e .
python download_mujoco_dataset.py
```