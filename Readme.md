```
project
|   README.md
|   run.py    
|---configs
|   |---dqn
|   |   |   config.yaml
|   |   |---agent
|   |   |   |   dqn.yaml
|   |   |---env
|   |   |   |   atari.yaml
|   |   |   |   procgen.yaml
|   |   |---model
|   |   |   |   nature.yaml
|   |   |   |   impala.yaml
|   |   |   |   resnet.yaml
|   |   |   |   ...
|   |   |--- ...
|   |---rainbow
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

### Downloading Dataset
#### ATARI
```
pip install gsutil
export data_dir = 'custom_path'
bash download_atari_replay_data.sh
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

git clone https://github.com/rail-berkeley/d4rl.git
cd d4rl
pip install -e .
python download_mujoco_dataset.py
```