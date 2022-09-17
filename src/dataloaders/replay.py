import gzip
import re
import os
from pathlib import Path
from typing import List, Tuple

import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from .base import BaseLoader
from src.envs.atari import AtariEnv
from src.common.data_utils import *


class ReplayDataset(Dataset):
    def __init__(self, 
                 data_type: str,
                 data_path: Path,
                 tmp_data_path: Path,
                 game: str,
                 checkpoint: int,
                 t_step: int,
                 max_size: int,
                 full_action_set: bool,
                 dataset_on_gpu: bool,
                 dataset_on_disk: bool,
                 device: str) -> None:

        device = torch.device(device)
        data = []
        self.dataset_on_disk = dataset_on_disk
        assert not (dataset_on_disk and dataset_on_gpu)
        filetypes = ['observation', 'action', 'reward', 'terminal']
        for i, filetype in enumerate(filetypes):
            filename = Path(data_path + '/' + f'{game}/{filetype}_{checkpoint}.gz')
            print(f'Loading {filename}')
                        
            # generate .npy data for obs for fast mmap_read
            if filetype == 'observation':
                new_filename = tmp_data_path + '/' + game
                new_filename = os.path.join(new_filename, Path(os.path.basename(filename)[:-3]+".npy"))
                try:
                    data_ = np.load(new_filename, mmap_mode="r+")
                except:
                    g = gzip.GzipFile(filename=filename)
                    data__ = np.load(g)
                    data___ = np.copy(data__[:max_size])
                    print(f'Using {data___.size * data___.itemsize} bytes')
            
                    np.save(new_filename, data___,)    
                    print("Stored on disk at {}".format(new_filename))

                    del data___
                    del data__
                    data_ = np.load(new_filename) #, mmap_mode="r+")
            
            # just load data for action, reward, and terminal
            else:
                g = gzip.GzipFile(filename=filename)
                data__ = np.load(g)

                # number of interactions for each checkpoint
                data___ = np.copy(data__[:max_size])
                print(f'Using {data___.size * data___.itemsize} bytes')
                del data__
                data_ = torch.from_numpy(data___)                            
            
            if ((filetype == 'action') and (data_type == 'atari') and (full_action_set)):
                action_mapping = dict(zip(data_.unique().numpy(),
                                          AtariEnv(re.sub(r'(?<!^)(?=[A-Z])', '_', game).lower()).ale.getMinimalActionSet()))
                data_.apply_(lambda x: action_mapping[x])
                
            if dataset_on_gpu:
                print("Stored on GPU")
                data_ = data_.to(device) #cuda(non_blocking=True).to(device)
                del data___
            data.append(data_)
            setattr(self, filetype, data_)
        
        self.game = game
        self.t = t_step
        self.size = min(self.action.shape[0], max_size)
        self.effective_size = (self.size - self.t + 1)

    def __len__(self) -> int:
        return self.effective_size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        time_ind = index % self.effective_size
        sl = slice(time_ind, time_ind+self.t)
        if self.dataset_on_disk:
            obs = torch.from_numpy(self.observation[sl])
        else:
            obs = (self.observation[sl])

        return tuple([obs,
                     self.action[sl],
                     self.reward[sl],
                     self.terminal[sl]])


class MultiReplayDataset(Dataset):
    def __init__(self, 
                data_type: str,
                data_path: Path,
                tmp_data_path: Path, 
                game: str,
                checkpoints: List[int],
                t_step: int,
                max_size: int,
                full_action_set: bool,
                dataset_on_gpu: bool,
                dataset_on_disk: bool,
                device: str) -> None:
        
        self.datasets =[ReplayDataset(data_type,
                                      data_path,
                                      tmp_data_path,
                                      game,
                                      ckpt,
                                      t_step,
                                      max_size,
                                      full_action_set,
                                      dataset_on_gpu,
                                      dataset_on_disk,
                                      device) for ckpt in checkpoints]

        self.num_blocks = len(self.datasets)
        self.block_len = len(self.datasets[0])

    def __len__(self) -> int:
        return len(self.datasets) * len(self.datasets[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ckpt_index = index % len(self.datasets)
        index = index // len(self.datasets)
        return self.datasets[ckpt_index][index]


class ReplayDataLoader(BaseLoader):
    name = 'replay'
    def __init__(self,
                 data_type: str,
                 data_path: Path,
                 tmp_data_path: Path,
                 game: str,
                 checkpoints: List[int],
                 t_step: int, # length of the trajectory to predict
                 max_size: int,
                 dataset_on_gpu: bool,
                 dataset_on_disk: bool,
                 batch_size: int,
                 full_action_set: bool,
                 num_workers: int,
                 pin_memory: bool,
                 prefetch_factor: int,
                 device: str,
                 group_read_factor: int=0,
                 shuffle_checkpoints: bool=False):
        
        super().__init__()
        self.data_type = data_type
        self.data_path = data_path
        self.tmp_data_path = tmp_data_path
        self.game = game
        self.checkpoints = checkpoints
        self.t_step = t_step
        self.max_size = max_size
        self.dataset_on_gpu = dataset_on_gpu
        self.dataset_on_disk = dataset_on_disk
        self.batch_size = batch_size
        self.full_action_set = full_action_set
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.device = device
        self.group_read_factor = group_read_factor
        self.shuffle_checkpoints = shuffle_checkpoints

    def get_dataloader(self):
        def collate(batch):
            """
            [params] observation 
                (atari): (N, T, H, W) 
                (dmc): (N, T, C, H, W)
            [returns] observation: (N, T, C, H, W) C=1 in atari
            """
            t_step = self.t_step
            observation, action, reward, done = torch.utils.data.dataloader.default_collate(batch)

            # grey-scale image for atari
            if self.data_type == 'atari':
                observation = observation.unsqueeze(2)
            
            action = action.long()
            reward = torch.nan_to_num(reward).sign()
            done = done.bool()
            
            # when done is True, func sanitize batch zeros out observation and reward
            return sanitize_batch(OfflineSamples(observation, action, reward, done))

        dataset = MultiReplayDataset(self.data_type,
                                    self.data_path, 
                                    self.tmp_data_path, 
                                    self.game, 
                                    self.checkpoints, 
                                    self.t_step, 
                                    self.max_size,
                                    self.full_action_set, 
                                    self.dataset_on_gpu, 
                                    self.dataset_on_disk,
                                    self.device)

        if self.shuffle_checkpoints:
            data = get_from_dataloaders(dataset.datasets)
            shuffled_data = shuffle_batch_dim(*data)
            assign_to_dataloaders(dataset.datasets, *shuffled_data)

        if self.group_read_factor != 0:
            sampler = CacheEfficientSampler(dataset.num_blocks, dataset.block_len, self.group_read_factor)
            dataloader = DataLoader(dataset, 
                                    batch_size=self.batch_size,
                                    sampler=sampler,
                                    num_workers=self.num_workers,
                                    pin_memory=self.pin_memory,
                                    collate_fn=collate,
                                    drop_last=True,
                                    prefetch_factor=self.prefetch_factor)
        else:
            dataloader = DataLoader(dataset, 
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=self.num_workers,
                                    pin_memory=self.pin_memory,
                                    collate_fn=collate,
                                    drop_last=True,
                                    prefetch_factor=self.prefetch_factor)

        return dataloader
