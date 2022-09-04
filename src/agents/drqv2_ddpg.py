import warnings 
warnings.filterwarnings('ignore', category=DeprecationWarning)

from turtle import update
import ipdb
import gzip
import datetime
import io
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

from .base import BaseAgent
from src.common.train_utils import schedule, eval_mode, soft_update_params
from src.common.augmentation import RandomShiftsAug
from src.common.utils import set_global_seeds


from collections import namedtuple
EnvInfo = namedtuple("EnvInfo", ["game_score", "traj_done"])  # Define in env file.


class DrQV2Agent(BaseAgent):
    name='drqv2'
    def __init__(self, 
                 cfg,
                 device, 
                 train_env, 
                 eval_env, 
                 logger, 
                 buffer,
                 aug_func,
                 model,
                 video_recorder=None):
        # cfg-basic
        self.cfg = cfg
        self.device = device 
        self.train_env, self.eval_env = train_env, eval_env
        self.logger = logger
        self.video_recorder = video_recorder
        self.buffer = buffer 
        self.model = model.to(self.device)
        self.game = cfg.game
        # cfg-update 
        self.aug = RandomShiftsAug(pad=4)
        self.num_expl_steps = cfg.num_expl_steps
        self.num_seed_steps = cfg.num_seed_steps
        self.critic_target_tau = cfg.critic_target_tau
        self.update_every_steps = cfg.update_every_steps
        self.stddev_schedule = cfg.stddev_schedule
        self.stddev_clip = cfg.stddev_clip
        self.encoder_opt = self._build_optimizer(self.model.backbone.parameters(), cfg.optimizer)
        self.actor_opt = self._build_optimizer(self.model.policy.actor.parameters(), cfg.optimizer)
        self.critic_opt = self._build_optimizer(self.model.policy.critic.parameters(), cfg.optimizer)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.train_mode()
        self.critic_target.train()

        # FIXME: fop saving dataset
        self.base_dir = Path.cwd() / 'data'
        self.new_dir = self.base_dir / 'dmc' / self.game / datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        print('video save directory:', self.new_dir)
        if not self.new_dir.exists():
            self.new_dir.mkdir(exist_ok=False, parents=True)


    @property
    def encoder(self):
        return self.model.backbone

    @property
    def actor(self):
        return self.model.policy.actor

    @property
    def critic(self):
        return self.model.policy.critic

    @property
    def critic_target(self):
        return self.model.policy.critic_target

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    def train_mode(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def _build_optimizer(self, parameters, optimizer_cfg):
        return optim.Adam(parameters, **optimizer_cfg)

    def predict(self, obs, sample=False):
        std = schedule(self.stddev_schedule, self.global_step)
        action = self.model.forward(obs, std, sample=sample)

        # exploration phase이면서 sampling(forwarding) 중일 때
        if (self.num_expl_steps > self.global_step) & (sample):
            action = np.random.uniform(-1,1,size=action.shape)

        return action.astype(np.float32)

    def update_critic(self, obs_b, act_b, rew_b, dis_b, next_obs_b):
        metrics = dict()
        with torch.no_grad():
            std = schedule(self.stddev_schedule, self.global_step)
            dist = self.actor(next_obs_b, std)
            next_act_b = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs_b, next_act_b)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = rew_b + (dis_b * target_V)

        Q1, Q2 = self.critic(obs_b, act_b)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = Q1.mean().item()
        metrics['critic_q2'] = Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        # encoder is only updated by critic loss
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs_b):
        metrics = dict()
        std = schedule(self.stddev_schedule, self.global_step)
        dist = self.actor(obs_b, std)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs_b, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_logprob'] = log_prob.mean().item()
        metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
        # ipdb.()

        return metrics
    

    # FIXME:
    def to_torch(self, xs, device):
        return tuple(torch.as_tensor(x, device=device) for x in xs)

    def update(self):
        metrics = dict()

        # idxes, obs_b, act_b, rew_b, done_b, gamma_b, next_obs_b, _ = self.buffer.sample(self.cfg.batch_size, ddpg=True)
        batch = self.buffer.sample(self.cfg.batch_size)
        obs_b, act_b, rew_b, dis_b, next_obs_b = self.to_torch(batch, self.device)

        N, S, C, W, H = obs_b.shape
        obs_b = obs_b.view(N, S*C, H, W)
        next_obs_b = next_obs_b.view(N, S*C, H, W)

        # augment
        obs_b = self.aug(obs_b.float())
        next_obs_b = self.aug(next_obs_b.float())

        obs_b = obs_b / 255.0 - 0.5
        next_obs_b = next_obs_b / 255.0 - 0.5

        # encode
        obs_b = self.encoder(obs_b)
        with torch.no_grad(), eval_mode(self):
            next_obs_b = self.encoder(next_obs_b)

        # update critic
        metric_critic = self.update_critic(obs_b, 
                                           act_b,
                                           rew_b,
                                           dis_b,
                                           next_obs_b)

        # update actor
        metric_actor = self.update_actor(obs_b.detach())

        # update critic target
        soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        metrics.update(metric_critic)
        metrics.update(metric_actor)
        return metrics


    def train(self):

        save_start_index = 0
        self._global_step = 0
        self._global_episode = 0
        obs = self.train_env.reset() 
        training_timesteps = (self.cfg.num_timesteps) // 2

        for t in tqdm.tqdm(range(training_timesteps)):

            obs_tensor = self.buffer.encode_obs(obs, prediction=True, ddpg=True) 

            # forward
            with torch.no_grad(), eval_mode(self):
                action = self.predict(obs_tensor, sample=True)

            next_obs, reward, done, info = self.train_env.step(action) 
            self.buffer.store(obs, action, reward, done) 
            self._global_step = t

            # update
            if (t >= self.num_seed_steps) & (t % self.cfg.update_every_steps ==0):
                log_data = self.update()
                self.logger.update_log(mode='train', **log_data)

            # logger
            info = EnvInfo(game_score=reward, traj_done=done)
            self.logger.step(obs, reward, done, info, mode='train')
            if t % self.cfg.log_every == 0:
                self.logger.write_log()

            # evaluate & save model 
            if t % self.cfg.eval_every == 0:
                self.logger.save_state_dict(model=self.model)
                self.evaluate()
            

            # move on
            if done:  
                self._global_episode += 1
                print(f'global_episode: {self._global_episode}')
                obs = self.train_env.reset() 

                # save_dataset
                # 한 episode에 500 step
                # 즉 250,000 step에 500 global_episode
                if self.global_episode in [4, 500, 1000, 1500, 2000, 2500, 3000]:

                    curr_global_frame = self.global_frame + 2
                    save_end_index = self.buffer.index % self.buffer.buffer_size

                    obs_name = self.new_dir / f'observation_{curr_global_frame/100000}'
                    act_name = self.new_dir / f'action_{curr_global_frame/100000}'
                    rew_name = self.new_dir / f'reward_{curr_global_frame/100000}'
                    done_name = self.new_dir / f'terminal_{curr_global_frame/100000}'
                    
                    # FIXME: 다 커버 가능한 logic은 아님 (buffer_size가 dataset 저장하는 간격(위에선500)보다 커야함)
                    if save_end_index < save_start_index:
                        save_valid_indices_1 = self.buffer.save_valid[save_start_index:]
                        save_valid_indices_2 = self.buffer.save_valid[:save_end_index]

                        _o1 = self.buffer.obs[save_start_index:][save_valid_indices_1]
                        _a1 = self.buffer.act[save_start_index:][save_valid_indices_1]
                        _r1 = self.buffer.rew[save_start_index:][save_valid_indices_1]
                        _d1 = self.buffer.done[save_start_index:][save_valid_indices_1]

                        _o2 = self.buffer.obs[:save_end_index][save_valid_indices_2]
                        _a2 = self.buffer.act[:save_end_index][save_valid_indices_2]
                        _r2 = self.buffer.rew[:save_end_index][save_valid_indices_2]
                        _d2 = self.buffer.done[:save_end_index][save_valid_indices_2]

                        _o = np.vstack((_o1, _o2))
                        _a = np.vstack((_a1, _a2))
                        _r = np.hstack((_r1, _r2))
                        _d = np.hstack((_d1, _d2))

                        # action-npz
                        self.save_data(_a, Path(str(act_name)+'.npz'))
                        # action-gz
                        f_action = gzip.GzipFile(str(act_name)+'.gz', "w")
                        np.save(f_action, _a)
                        f_action.close()
                        print('action 저장 완료')
                        # reward-npz
                        self.save_data(_r, Path(str(rew_name)+'.npz'))
                        # reward-gz
                        f_reward = gzip.GzipFile(str(rew_name)+'.gz', "w")
                        np.save(f_reward, _r)
                        f_reward.close()
                        print('reward 저장 완료')
                        # terminal-npz
                        self.save_data(_d, Path(str(done_name)+'.npz'))
                        # teminal-gz
                        f_done = gzip.GzipFile(str(done_name)+'.gz', "w")
                        np.save(f_done, _d)
                        f_done.close()
                        print('terminal 저장 완료')
                        # observation-npz
                        self.save_data(_o, Path(str(obs_name)+'.npz'))
                        # observation-gz
                        f_obs = gzip.GzipFile(str(obs_name)+'.gz', "w")
                        np.save(f_obs, _o)
                        f_obs.close()
                        print('obs 저장 완료')

                        del _o1, _o2, _a1, _a2, _r1, _r2, _d1, _d2


                    else:
                        save_valid_indices = self.buffer.save_valid[save_start_index:save_end_index]
                        _o1 = self.buffer.obs[save_start_index:save_end_index][save_valid_indices]
                        _a1 = self.buffer.act[save_start_index:save_end_index][save_valid_indices]
                        _r1 = self.buffer.rew[save_start_index:save_end_index][save_valid_indices]
                        _d1 = self.buffer.done[save_start_index:save_end_index][save_valid_indices]


                        # action-npz
                        self.save_data(_a1, Path(str(act_name)+'.npz'))
                        # action-gz
                        f_action = gzip.GzipFile(str(act_name)+'.gz', "w")
                        np.save(f_action, _a1)
                        f_action.close()
                        print('action 저장 완료')
                        # reward-npz
                        self.save_data(_r1, Path(str(rew_name)+'.npz'))
                        # reward-gz
                        f_reward = gzip.GzipFile(str(rew_name)+'.gz', "w")
                        np.save(f_reward, _r1)
                        f_reward.close()
                        print('reward 저장 완료')
                        # terminal-npz
                        self.save_data(_d1, Path(str(done_name)+'.npz'))
                        # teminal-gz
                        f_done = gzip.GzipFile(str(done_name)+'.gz', "w")
                        np.save(f_done, _d1)
                        f_done.close()
                        print('terminal 저장 완료')
                        # observation-npz
                        self.save_data(_o1, Path(str(obs_name)+'.npz'))
                        # observation-gz
                        f_obs = gzip.GzipFile(str(obs_name)+'.gz', "w")
                        np.save(f_obs, _o1)
                        f_obs.close()
                        print('obs 저장 완료')

                        del _o1, _a1, _r1, _d1

                    save_start_index = save_end_index

            else: 
                obs = next_obs

            
    def save_data(self, data, dir):
        with io.BytesIO() as bs:
            np.savez_compressed(bs, data)
            bs.seek(0)
            with dir.open('wb') as f:
                f.write(bs.read())



    def evaluate(self):
        self.model.eval()
        for idx in tqdm.tqdm(range(self.cfg.num_eval_trajectories)):
            obs = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(idx == 0))
            while True:                
                obs_tensor = self.buffer.encode_obs(obs, prediction=True, ddpg=True)
                # get action from the model
                with torch.no_grad(), eval_mode(self):
                    action = self.predict(obs_tensor, sample=False)
                # step
                next_obs, reward, done, info = self.eval_env.step(action)
                # logger
                info = EnvInfo(game_score=reward, traj_done=done)
                self.logger.step(obs, reward, done, info, mode='eval')
                self.video_recorder.record(self.eval_env)
                # move on
                if done:
                    self.logger.write_log(mode='eval')
                    self.video_recorder.save(f'{self.global_frame}_{idx}.mp4')
                    break
                else:
                    obs = next_obs

        self.logger.write_log(mode='eval')


