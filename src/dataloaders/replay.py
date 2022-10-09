import gzip
import re
import os
from pathlib import Path
from typing import List, Tuple

import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from .base import BaseLoader
from src.envs.atari import AtariEnv
from src.common.data_utils import *
from einops import rearrange


class ReplayDataset(Dataset):
    def __init__(self, 
                 data_type: str,
                 data_path: Path,
                 tmp_data_path: Path,
                 game: str,
                 run: int,
                 checkpoint: int,
                 frame: int,
                 t_step: int,
                 max_size: int,
                 minimal_action_set: bool,
                 dataset_on_gpu: bool,
                 dataset_on_disk: bool,
                 device: str) -> None:

        device = torch.device(device)
        data = []
        self.dataset_on_disk = dataset_on_disk
        assert not (dataset_on_disk and dataset_on_gpu)
        filetypes = ['observation', 'action', 'reward', 'terminal']
        for i, filetype in enumerate(filetypes):
            filename = Path(data_path + '/' + f'{game}/{filetype}_{run}_{checkpoint}.gz')
            print(f'Loading {filename}')
                        
            # generate .npy data for obs for fast mmap_read
            if filetype == 'observation':
                new_filename = tmp_data_path + '/' + game
                new_filename = os.path.join(new_filename, Path(os.path.basename(filename)[:-3]+".npy"))
                try:
                    if dataset_on_disk:
                        data_ = np.load(new_filename, mmap_mode="r+")
                    if dataset_on_gpu:
                        data__ = np.load(new_filename)
                        data_ = torch.from_numpy(data__)
                except:
                    g = gzip.GzipFile(filename=filename)
                    data__ = np.load(g)
                    data___ = np.copy(data__[:max_size])
                    print(f'Using {data___.size * data___.itemsize} bytes')
            
                    np.save(new_filename, data___,)    
                    print("Stored on disk at {}".format(new_filename))

                    del data___
                    del data__
                    data_ = np.load(new_filename, mmap_mode="r+")
            
            # just load data for action, reward, and terminal
            else:
                g = gzip.GzipFile(filename=filename)
                data__ = np.load(g)

                # number of interactions for each checkpoint
                data___ = np.copy(data__[:max_size])
                print(f'Using {data___.size * data___.itemsize} bytes')
                del data__
                data_ = torch.from_numpy(data___)                            
            
            if ((filetype == 'action') and (data_type == 'atari') and (not minimal_action_set)):
                action_mapping = dict(zip(data_.unique().numpy(),
                                          AtariEnv(game).ale.getMinimalActionSet()))
                data_.apply_(lambda x: action_mapping[x])
                                
            if dataset_on_gpu:
                print("Stored on GPU")
                data_ = data_.to(device) #cuda(non_blocking=True).to(device)
                
            data.append(data_)
            setattr(self, filetype, data_)
        
        self.game = game
        self.f = frame
        self.t = t_step
        self.size = min(self.action.shape[0], max_size)
        self.effective_size = (self.size - self.f - self.t + 1)

    def __len__(self) -> int:
        return self.effective_size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        time_ind = index % self.effective_size
        sl = slice(time_ind, time_ind + self.t + (self.f-1))
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
                runs: List[int],
                checkpoints: List[int],
                frame: int,
                t_step: int,
                max_size: int,
                minimal_action_set: bool,
                dataset_on_gpu: bool,
                dataset_on_disk: bool,
                device: str) -> None:
        
        datasets = []
        for run in runs:
            for ckpt in checkpoints:
                datasets.append(ReplayDataset(data_type,
                                              data_path,
                                              tmp_data_path,
                                              game,
                                              run,
                                              ckpt,
                                              frame,
                                              t_step,
                                              max_size,
                                              minimal_action_set,
                                              dataset_on_gpu,
                                              dataset_on_disk,
                                              device))
        self.datasets = datasets
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
                 runs: List[int],
                 checkpoints: List[int],
                 frame: int,
                 t_step: int, # length of the trajectory to predict
                 max_size: int,
                 dataset_on_gpu: bool,
                 dataset_on_disk: bool,
                 batch_size: int,
                 minimal_action_set: bool,
                 num_workers: int,
                 pin_memory: bool,
                 prefetch_factor: int,
                 device: str,
                 shuffle_checkpoints: bool,
                 shuffle: bool):
        
        super().__init__()
        self.data_type = data_type
        self.data_path = data_path
        self.tmp_data_path = tmp_data_path
        self.game = game
        self.runs = runs
        self.checkpoints = checkpoints
        self.frame = frame
        self.t_step = t_step
        self.max_size = max_size
        self.dataset_on_gpu = dataset_on_gpu
        self.dataset_on_disk = dataset_on_disk
        self.batch_size = batch_size
        self.minimal_action_set = minimal_action_set
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.device = device
        self.shuffle_checkpoints = shuffle_checkpoints
        self.shuffle = shuffle
        
    def get_dataloader(self):
        def collate(batch):
            """
            [params] observation 
                (atari): (n, t, h, w) 
                (dmc): (n, t, c, h, w)
            [returns] observation: (n, t, f*c, h, w) c=1 in atari, c=3 in dmc
            """
            f = self.frame
            observation, action, reward, done = torch.utils.data.dataloader.default_collate(batch)
                          
            # grey-scale image for atari
            if self.data_type == 'atari':
                observation = rearrange(observation, 'n t h w -> n t 1 h w')
            
            # process data-format
            observation = rearrange(observation, 'n t c h w -> n t 1 c h w')
            observation = observation.repeat(1, 1, f, 1, 1, 1)
            action = action.long()
            reward = torch.nan_to_num(reward).sign()
            done = done.bool()
            
            # frame-stack
            if f != 1:
                for i in range(1, f):
                    observation[:, :, i] = observation[:, :, i].roll(-i, 1)
                observation= observation[:, :-(f-1)]
                action = action[:, f-1:]
                reward = reward[:, f-1:]
                done = done[:, f-1:]
            
            # when done is True, func sanitize batch zeros out observation and reward
            return sanitize_batch(OfflineSamples(observation, action, reward, done))

        dataset = MultiReplayDataset(self.data_type,
                                    self.data_path, 
                                    self.tmp_data_path, 
                                    self.game, 
                                    self.runs,
                                    self.checkpoints, 
                                    self.frame,
                                    self.t_step, 
                                    self.max_size,
                                    self.minimal_action_set, 
                                    self.dataset_on_gpu, 
                                    self.dataset_on_disk,
                                    self.device)

        if self.shuffle_checkpoints:
            data = get_from_dataloaders(dataset.datasets)
            shuffled_data = shuffle_batch_dim(*data)
            assign_to_dataloaders(dataset.datasets, *shuffled_data)

        dataloader = DataLoader(dataset, 
                                batch_size=self.batch_size,
                                shuffle=self.shuffle,
                                num_workers=self.num_workers,
                                pin_memory=self.pin_memory,
                                collate_fn=collate,
                                drop_last=False,
                                prefetch_factor=self.prefetch_factor)

        return dataloader
