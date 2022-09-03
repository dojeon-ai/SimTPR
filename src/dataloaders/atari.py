import gzip
import re
import os
from pathlib import Path
from typing import List, Tuple
from itertools import zip_longest

import tqdm
import numpy as np
import skimage as sk
import torch
import cv2
from torch.utils.data import DataLoader, Dataset
from .base import BaseLoader
from src.envs.atari import AtariEnv
from src.common.data_utils import *


class DQNReplayDataset(Dataset):
    def __init__(self, 
                 data_path: Path,
                 tmp_data_path: Path,
                 game: str,
                 checkpoint: int,
                 frames: int,
                 t_step: int,
                 max_size: int,
                 full_action_set: bool,
                 dataset_on_gpu: bool,
                 dataset_on_disk: bool,
                 device: str,
                 is_dmc: str) -> None:

        device = torch.device(device)
        data = []
        self.dataset_on_disk = dataset_on_disk
        assert not (dataset_on_disk and dataset_on_gpu)
        filetypes = ['observation', 'action', 'reward', 'terminal', 'flow', 'hog']
        for i, filetype in enumerate(filetypes):
            filename = Path(data_path + '/' + f'{game}/{filetype}_{checkpoint}.gz')
            print(f'Loading {filename}')
            
            # generate flow data if not exists
            if filetype == 'flow':
                try:
                    g = gzip.GzipFile(filename=filename)
                except:
                    print(f'optical flow data does not exist. estimate the optical flows.')
                    flows = []
                    T = self.observation.shape[0]
                    for t in tqdm.tqdm(range(T-1)):
                        _prev = np.expand_dims(self.observation[t], -1)
                        _next = np.expand_dims(self.observation[t+1], -1)

                        # get flow-map
                        # this parameters are tuned for ATARI games
                        flow = cv2.calcOpticalFlowFarneback(prev=_prev,
                                                            next=_next,
                                                            flow=None,
                                                            pyr_scale=0.5,
                                                            levels=3, 
                                                            winsize=6,
                                                            iterations=20,
                                                            poly_n=7,
                                                            poly_sigma=1.5,
                                                            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN) 
                        flows.append(flow.astype(np.float16))   

                    # flow at last timestep is stored as zero
                    flows.append(np.zeros_like(flow).astype(np.float16))
                    flows = np.stack(flows)
                    g = gzip.GzipFile(filename, "w")
                    np.save(file=g, arr=flows)
                    g.close()
                        
            # generate hog data if not exists
            if filetype == 'hog':
                try:
                    g = gzip.GzipFile(filename=filename)
                except:
                    print(f'hog data does not exist. compute the hog feature.')
                    hogs = []
                    T = self.observation.shape[0]
                    for t in tqdm.tqdm(range(T)):
                        img = np.expand_dims(self.observation[t], -1)

                        # get hog
                        hog = sk.feature.hog(img, 
                                             orientations=9, 
                                             pixels_per_cell=(16, 16),
                                             cells_per_block=(3, 3), 
                                             visualize=False, 
                                             multichannel=True)
                        
                        hogs.append(hog.astype(np.float16))   

                    hogs = np.stack(hogs)
                    g = gzip.GzipFile(filename, "w")
                    np.save(file=g, arr=hogs)
                    g.close()
                        
            # generate .npy data for obs and flow for fast mmap_read
            if filetype in ['observation', 'flow', 'hog']:
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
            
            if ((filetype == 'action') and full_action_set) and (not is_dmc):
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
        self.f = frames
        self.t = t_step
        self.size = min(self.action.shape[0], max_size)
        self.effective_size = (self.size - self.f - self.t + 1)

    def __len__(self) -> int:
        return self.effective_size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
                                               torch.Tensor, torch.Tensor]:
        # batch_ind = index // self.effective_size
        time_ind = index % self.effective_size
        sl = slice(time_ind, time_ind+self.f+self.t)
        if self.dataset_on_disk:
            obs = torch.from_numpy(self.observation[sl])
            flow = torch.from_numpy(self.flow[sl])
            hog = torch.from_numpy(self.hog[sl])
        else:
            obs = (self.observation[sl])
            flow = (self.flow[sl])
            hog = (self.hog[sl])

        return tuple([obs,
                     self.action[sl],
                     self.reward[sl],
                     self.terminal[sl],
                     flow,
                     hog])


class MultiDQNReplayDataset(Dataset):
    def __init__(self, 
                data_path: Path,
                tmp_data_path: Path, 
                game: str,
                checkpoints: List[int],
                frames: int,
                t_step: int,
                max_size: int,
                full_action_set: bool,
                dataset_on_gpu: bool,
                dataset_on_disk: bool,
                device: str,
                is_dmc: bool) -> None:
        
        # checkpoints: [50]
        self.datasets =[DQNReplayDataset(data_path,
                        tmp_data_path,
                        game,
                        ckpt,
                        frames,
                        t_step,
                        max_size,
                        full_action_set,
                        dataset_on_gpu,
                        dataset_on_disk,
                        device,
                        is_dmc) for ckpt in checkpoints]

        self.num_blocks = len(self.datasets)
        self.block_len = len(self.datasets[0])

    def __len__(self) -> int:
        return len(self.datasets) * len(self.datasets[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                               torch.Tensor, torch.Tensor, torch.Tensor]:
        ckpt_index = index % len(self.datasets)
        index = index // len(self.datasets)
        return self.datasets[ckpt_index][index]


class ATARILoader(BaseLoader):
    name = 'atari'
    def __init__(self,
                 data_path: Path,
                 tmp_data_path: Path,
                 game: str,
                 checkpoints: List[int],
                 frames: int,
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
        self.data_path = data_path
        self.tmp_data_path = tmp_data_path
        self.game = game
        self.checkpoints = checkpoints
        self.frames = frames
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
        if 'dmc' in self.data_path:
            self.is_dmc = True
        else:
            self.is_dmc = False

    def get_dataloader(self):
        def collate(batch):
            frames = self.frames
            # observation shape
            # ATARI: (B, self.f+self.t, H=84, W=84)
            # DMC: (B, self.f+self.t, C=3, H=84, W=84)
            observation, action, reward, done, flow, hog = torch.utils.data.dataloader.default_collate(batch)
            """
            # tbcfhw: standard format for ATC
            observation = torch.einsum('bthw->tbhw', observation).unsqueeze(2).repeat(1, 1, frames, 1, 1)
            for i in range(1, frames):
                observation[:, :, i] = observation[:, :, i].roll(-i, 0)
            observation = observation[:-frames] # do not use the last frame (just to make t=k)
            observation = observation.unsqueeze(3)
            action = torch.einsum('bt->tb', action)[frames-1:-1].long()
            reward = torch.einsum('bt->tb', reward)[frames-1:-1]
            reward = torch.nan_to_num(reward).sign()  # Apparently possible, somehow.
            done = torch.einsum('bt->tb', done)[frames-1:-1].bool()
            """
            if self.is_dmc:
                observation = observation.unsqueeze(2).repeat(1, 1, frames, 1, 1, 1)
            else:
                observation = observation.unsqueeze(2).repeat(1, 1, frames, 1, 1)
            for i in range(1, frames):
                observation[:, :, i] = observation[:, :, i].roll(-i, 1)
            # ATARI: b, t+f, f, h, w
            # DMC:   b, t+f, f, c, h, w
            observation = observation[:, :-frames] # do not use the last frame (just to make t=k)
            # ATARI: b, t, f, h, w
            # DMC:   b, t, f, c, h, w
            if not self.is_dmc:
                observation = observation.unsqueeze(3)
            # ATARI, DMC: b, t, f, c, h, w
            action = action[:, frames-1:-1].long()
            reward = reward[:, frames-1:-1]
            reward = torch.nan_to_num(reward).sign()  # Apparently possible, somehow.
            done = done[:, frames-1:-1].bool()
            # b, t, h, w, c -> b, t, c, h, w
            flow = flow[:, frames-1:-1].permute(0,1,4,2,3) 
            # b, t, h_d
            hog = hog[:, frames-1:-1]
            
            return sanitize_batch(OfflineSamples(observation, action, reward, done, flow, hog))

        dataset = MultiDQNReplayDataset(self.data_path, 
                                        self.tmp_data_path, 
                                        self.game, 
                                        self.checkpoints, 
                                        self.frames, 
                                        self.t_step, 
                                        self.max_size,
                                        self.full_action_set, 
                                        self.dataset_on_gpu, 
                                        self.dataset_on_disk,
                                        self.device,
                                        self.is_dmc)

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
