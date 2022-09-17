### 0. Update Notes (09.18)
```
1. [DataLoader]: t_step and frame_stack is integrated to one arugment.
   - (before) (n, t, f, c, h, w)
   - (after) (n, t, c, h, w) 
     n: batch_size
     t: time_step = frame_stack
     c: channel
     h: height
     w: width
    
2. [Model]: Trainer Model and Agent Model is separated.
   - Each model components' input & output shape is modified & fixed.
   - Each component always output information dictionary (for debugging or vis purpose).
     | TrainerModel:
        | init 
           | backbone: (n,t,d) -> (n,t,d), info
           | head: (n,t,d)->(n,t,d), info
        | forward(x)
           | x, info = backbone(x)
           | x, info = head(x)
           | return x, info
      | AgentModel:
        | init 
           | backbone: (n,t,d) -> (n,t,d), info
           | head: (n,t,d)->(n,t,d), info # used when SSL loss is applied in fine-tuning.
           | neck: (n,t,d)->(n,d), info # used to integrate temporal dynamics (e.g., LSTM).
           | policy: (n,d)->(n,), info
        | forward(x)
           | x, info = backbone(x)
           | x, info = neck(x)
           | x, info = policy(x)
           | return x, info
           
3. [Model & Trainer]: Argument process-type is introduced. 
   - process_type == stack_frame: backbone encode stacked frames.
      - backbone: (n,t,d) -> (n,1,d)
      - neck: (n,1,d)->(n,d)
   - process_type == indiv_frame: backbone encode frames individually.
      - backbone: (n, t, d) -> (n, t, d)
      - neck: (n,t,d)->(n,d) (e.g., LSTM, concat, pool) 
      
4. [Model]: (head & neck)'s input shape is not pre-defined. 
   def build_model(cfg):
     ...
     fake_obs = torch.zeros((1, *backbone_cfg['obs_shape']))
     out, _ = backbone(fake_obs)
     output_dim = out.shape[-1]
     
     head_cfg['in_features'] = output_dim
     neck_cfg['in_features'] = output_dim
     ...
    
5. [Trainer]: all trainers share base train function.
   - Each trainer is required to write its own loss computation and evaluation function.
       @abstractmethod
       def compute_loss(self, obs, act, rew, done) -> Tuple[torch.Tensor, dict]:
          pass
    
       @abstractmethod
       # custom model update other than backpropagation (e.g., ema)
       def update(self, obs, act, rew, done):
          pass
    
       @abstractmethod
       def evaluate(self, obs, act, rew, done) -> dict:
          pass
   
6. [Logger]: If you would like to log anything in pre-training (e.g., wandb.Image), 
     just put it inside to the dictionary of compute_loss or evaluate function.
     |class TrainerLogger:
         def __init__(self):
            self.average_meter_set = AverageMeterSet()
            self.media_set = {}
    
         def update_log(self, **kwargs):
            for k, v in kwargs.items():
               if isinstance(v, float) or isinstance(v, int):
                  self.average_meter_set.update(k, v)
               else:
                  self.media_set[k] = v
```

### 1. Project Structure
```
project
|   README.md
|   run_pretrain.py # pretrain a model with a given config
|   run_finetune.py # finetune a model with a given config
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
|   |   |   |   |   loader1.yaml
|   |   |   |   |   ...
|   |   |   |---env
|   |   |   |   |   env1.yaml
|   |   |   |   |   ...
|   |   |   |---model
|   |   |   |   |--- backbones
|   |   |   |   |   |   backbone1.yaml
|   |   |   |   |   |   ...
|   |   |   |   |--- heads
|   |   |   |   |   |   head1.yaml
|   |   |   |   |   |   ...
|   |   |   |---trainer
|   |   |   |   |   trainer1.yaml
|   |   |   |   |   ...
|   |   |---finetune
|   |   |   |  config.yaml
|   |   |   |---env
|   |   |   |   |   env1.yaml
|   |   |   |   |   ...
|   |   |   |---model
|   |   |   |   |--- backbones
|   |   |   |   |   |   backbone1.yaml
|   |   |   |   |   |   ...
|   |   |   |   |--- necks
|   |   |   |   |   |   neck1.yaml
|   |   |   |   |   |   ...
|   |   |   |   |--- policies
|   |   |   |   |   |   policy1.yaml
|   |   |   |   |   |   ...
|   |   |   |---agent
|   |   |   |   |   agent1.yaml
|   |   |   |   |   ...
|   |--- ...
|   
└───src
    |--- agents
    |--- common
    |--- dataloaders
    |--- envs
    |--- models
         |--- backbones
         |--- heads
    |--- trainers
```

### 2. Installation
#### ATARI
```
download_rom.ipynb
```
#### DMC (TBU)
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
apt-get install libegl1
```

### 3. Dataset
#### ATARI
```
pip install gsutil
export data_dir = 'custom_path'
bash download_atari_replay_data.sh
```

#### DMC (TBU)
```
```

### 4. TODO
```
1. Modify MLR (code & config) with (model & trainer)'s new API
2. Multiprocessing code to bash script (why? each PPO agent requires multiprocessing).
3. Save model's weights to a local folder not in an artifact.
4. Set environment's observation shape to (n,t,c,h,w).
5. Modify a base agent's API which can integrate Rainbow, DDPG, PPO.
6. Implement various necks. (e.g., LSTM, pool).
7. Integrate weight init & augmentation function.
8. Modify DMC with a new API.
9. Implement test-code to check the safety of a code-update.
10. Integrate PPO & ProcGen environment.
11. Check the performance of Rainbow + ATARI, DDPG + DMC, PPO + ProcGen.
```
