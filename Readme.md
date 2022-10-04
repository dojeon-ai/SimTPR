### 0. Update Notes for API (10.04)
```
1. [Model]: Trainer Model and Agent Model is separated.
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
           
2. [Model]: (head & neck)'s input shape is not pre-defined. 
   def build_model(cfg):
     ...
     fake_obs = torch.zeros((1, *backbone_cfg['obs_shape']))
     out, _ = backbone(fake_obs)
     output_dim = out.shape[-1]
     
     head_cfg['in_dim'] = output_dim
     neck_cfg['in_dim'] = output_dim
     ...
    
3. [Trainer]: all trainers share base train function.
   - Each trainer is required to write its own loss computation custom update function.
       @abstractmethod
       def compute_loss(self, obs, act, rew, done) -> Tuple[torch.Tensor, dict]:
          pass
    
       @abstractmethod
       # custom model update other than backpropagation (e.g., ema)
       def update(self, obs, act, rew, done):
          pass
    
4. [Trainer]: when pre-training, we use the identical evaluation protocol from Zhang et al.
   - Light-weight probing of unsupervised RL: https://arxiv.org/abs/2208.12345
     
     def evaluate(self) -> dict:
        eval_logs = {}
        rew_eval_logs = self.probe_reward()
        act_eval_logs = self.probe_action()
        eval_logs.update(rew_eval_logs)
        eval_logs.update(act_eval_logs)

        return eval_logs
   
5. [Trainer]: Append debugging option. (default: debug=True)
    When debug mode, we train data on a single batch.
     
    def debug(self):
        for batch in tqdm.tqdm(self.train_loader):   
            # forward
            loss, train_logs = self.compute_loss(obs, act, rew, done)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.update(obs, act, rew, done)

            break

        eval_logs = self.evaluate()
        
    - before merge or integration, you can test the various pre-trainer through scripts/test_pretrain.sh
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
1. Multiprocessing code to bash script (why? each PPO agent requires multiprocessing).
2. Save model's weights to a local folder not in an artifact.
3. Modify a base agent's API which can integrate Rainbow, DDPG, PPO.
4. Implement various necks. (e.g., LSTM, pool).
5. Integrate weight init & augmentation function.
6. Modify DMC with a new API.
7. Integrate PPO & ProcGen environment.
8. Check the performance of Rainbow + ATARI, DDPG + DMC, PPO + ProcGen.
```


### 5. Done
```
1. Implement RSSM (architecture from Light-weight probing & SGI)
2. Implement MLR (can be viewed as a BERT version of RSSM)
```