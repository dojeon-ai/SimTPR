from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F
import copy
import tqdm

    
class Beam():
    def __init__(self, 
                 model, 
                 device,
                 num_envs,
                 action_size, 
                 rtg_scale,
                 beam_width,
                 horizon,
                 context_len,
                 gamma):
        
        self.model = model
        self.device = device
        self.num_envs = num_envs
        self.action_size = action_size
        self.rtg_scale = rtg_scale
        
        self.beam_width = beam_width
        self.horizon = horizon
        self.context_len = context_len
        self.gamma = gamma
        
        
    @torch.no_grad()
    def rollout(self, state):
        """
        [param] state
            obs_list: (n, t, d)
            act_list: (n, t)
            rew_list: (n, t)
            rtg_list: (n, t)
        beam
            obs_batch: (n, b, t, d)
            act_batch: (n, b, t)
            rew_batch: (n, b, t)
            rtg_batch: (n, b, t)
        [return] act_sequence
        [return] rtg_sequence
        """
        # initialize beam
        beams, likelihoods = self._initialize_beams(state)
        
        # search the probable beams
        for _ in range(self.horizon):
            candidates, c_likelihoods = self._generate_candidates(beams, likelihoods)
            beams, likelihoods = self._select_candidates(candidates, c_likelihoods)
            beams = self._generate_sequence(beams)
        
        # select the beam with highest expected return
        beam = self._select_best_beam(beams)
        
        return beam
    
    
    def _initialize_beams(self, state):
        _state = copy.deepcopy(state)
        N = self.num_envs
        B = self.beam_width
        C = self.context_len
        PAD = 0
        
        # PAD
        _state['act_list'].append([PAD] * self.num_envs)
        _state['rew_list'].append([PAD] * self.num_envs)
        _state['rtg_list'].append([PAD] * self.num_envs)
        
        # stack and extract context
        obs_list = list(_state['obs_list'])
        act_list = np.stack(_state['act_list'], 1)
        rew_list = np.stack(_state['rew_list'], 1)
        rtg_list = np.stack(_state['rtg_list'], 1)

        obs_batch = torch.cat(obs_list, 1)
        act_batch = torch.LongTensor(act_list)
        rew_batch = torch.FloatTensor(rew_list)
        rtg_batch = torch.FloatTensor(rtg_list)
        
        obs_batch = obs_batch[:, -C:]
        act_batch = act_batch[:, -C:]
        rew_batch = rew_batch[:, -C:]
        rtg_batch = rtg_batch[:, -C:]
        
        # init non-used beams
        obs_batch = obs_batch.unsqueeze(1).repeat(1, B, 1, 1)
        act_batch = act_batch.unsqueeze(1).repeat(1, B, 1)
        rew_batch = rew_batch.unsqueeze(1).repeat(1, B, 1)
        rtg_batch = rtg_batch.unsqueeze(1).repeat(1, B, 1)
        
        inf = 1e9
        likelihoods = [0.0] + [-inf] * (B-1)
        likelihoods = torch.FloatTensor(likelihoods)
        likelihoods = likelihoods.unsqueeze(0).repeat(N, 1)
        
        beams = {
            'obs_batch': obs_batch,
            'act_batch': act_batch,
            'rew_batch': rew_batch,
            'rtg_batch': rtg_batch
        }
        return beams, likelihoods
        
        
    def _generate_candidates(self, beams, likelihoods):
        """
        [params] beams: (N, B, C, x)
        [params] likelihoods: (N, B)
        
        [returns] candidates:  (N, B*A, C, x)
        [returns] c_likelihoods: (N, B*A)
        """
        A = self.action_size
        N = self.num_envs
        B = self.beam_width
        C = self.context_len
        device = self.device
        
        obs_batch = rearrange(beams['obs_batch'], 'n b t d -> (n b) t d')
        act_batch = rearrange(beams['act_batch'], 'n b t -> (n b) t')
        rew_batch = rearrange(beams['rew_batch'], 'n b t -> (n b) t')
        rtg_batch = rearrange(beams['rtg_batch'], 'n b t -> (n b) t')
        
        ###############################################
        # (1) get action probability
        dec_input = {
            'obs': obs_batch[:, -C:].to(device),
            'act': act_batch[:, -C:].to(device),
            'rew': rew_batch[:, -C:].to(device),
            'rtg': rtg_batch[:, -C:].to(device)
        }
        dec_output = self.model.head.decode(dec_input, dataset_type='trajectory')
        probs = F.softmax(dec_output['act'], -1)[:, -1, :]
        
        ###############################################
        # (2) generate candidates
        obs_batch = torch.repeat_interleave(obs_batch, A, 0)
        act_batch = torch.repeat_interleave(act_batch, A, 0)
        rew_batch = torch.repeat_interleave(rew_batch, A, 0)
        rtg_batch = torch.repeat_interleave(rtg_batch, A, 0)
        
        actions = torch.arange(A).repeat(B).repeat(N)
        act_batch[:, -1] = actions
        
        obs_batch = rearrange(obs_batch, '(n b a) t d -> n (b a) t d', n=N, b=B, a=A)
        act_batch = rearrange(act_batch, '(n b a) t -> n (b a) t', n=N, b=B, a=A)
        rew_batch = rearrange(rew_batch, '(n b a) t -> n (b a) t', n=N, b=B, a=A)
        rtg_batch = rearrange(rtg_batch, '(n b a) t -> n (b a) t', n=N, b=B, a=A)
        
        candidates = {
            'obs_batch': obs_batch,
            'act_batch': act_batch,
            'rew_batch': rew_batch,
            'rtg_batch': rtg_batch
        }
        
        probs = torch.log(probs)        
        probs = rearrange(probs, '(n b) a -> n (b a)', n=N, b=B, a=A)
        probs = probs.cpu()

        likelihoods = torch.repeat_interleave(likelihoods, A, 1)
        likelihoods += probs
        
        return candidates, likelihoods
    
    
    def _select_candidates(self, candidates, c_likelihoods):
        """
        [params] candidates:  (N, B*A, C, x)
        [params] c_likelihoods: (N, B*A)
        
        [returns] beams:  (N, B, C, x)
        [returns] likelihoods: (N, B)
        """
        B = self.beam_width
        _, _, C, D = candidates['obs_batch'].shape
        
        _, top_B_indices = torch.topk(c_likelihoods, k=B, dim=1)
        
        top_B_indices = top_B_indices.unsqueeze(-1).repeat(1, 1, C)
        top_B_obs_indices = top_B_indices.unsqueeze(-1).repeat(1, 1, 1, D)
        
        obs_batch = torch.gather(candidates['obs_batch'], 1, top_B_obs_indices)
        act_batch = torch.gather(candidates['act_batch'], 1, top_B_indices)
        rew_batch = torch.gather(candidates['rew_batch'], 1, top_B_indices)
        rtg_batch = torch.gather(candidates['rtg_batch'], 1, top_B_indices)
        
        beams = {
            'obs_batch': obs_batch,
            'act_batch': act_batch,
            'rew_batch': rew_batch,
            'rtg_batch': rtg_batch
        }
        likelihoods = torch.gather(c_likelihoods, 1, top_B_indices[:, :, 0])
        
        return beams, likelihoods
    
        
    def _generate_sequence(self, beams):
        """
        [params] beams: (N, B, C, x)
        [returns] beams: (N, B, C, x)
        """
        A = self.action_size
        B = self.beam_width
        C = self.context_len
        N = self.num_envs
        device = self.device
        _, _, _, D = beams['obs_batch'].shape
        
        obs_batch = rearrange(beams['obs_batch'], 'n b t d -> (n b) t d')
        act_batch = rearrange(beams['act_batch'], 'n b t -> (n b) t')
        rew_batch = rearrange(beams['rew_batch'], 'n b t -> (n b) t')
        rtg_batch = rearrange(beams['rtg_batch'], 'n b t -> (n b) t')
        
        ##########################################
        # (1) generate reward for beams
        dec_input = {
            'obs': obs_batch[:, -C:].to(device),
            'act': act_batch[:, -C:].to(device),
            'rew': rew_batch[:, -C:].to(device),
            'rtg': rtg_batch[:, -C:].to(device)
        }
        dec_output = self.model.head.decode(dec_input, dataset_type='trajectory')
        rews = dec_output['rew']
        
        # <fill> last reward with predicted reward
        rew = rews[: ,-1, :].squeeze(-1)
        rew_batch[:, -1] = rew
        
        ###########################################
        # (2) generate rtg with predicted rewards
        dec_input = {
            'obs': obs_batch[:, -C:].to(device),
            'act': act_batch[:, -C:].to(device),
            'rew': rew_batch[:, -C:].to(device),
            'rtg': rtg_batch[:, -C:].to(device)
        }
        dec_output = self.model.head.decode(dec_input, dataset_type='trajectory')
        rtgs = dec_output['rtg']
        
        # <fill> last rtg with predicted rtg
        rtg = rtgs[: ,-1, :].squeeze(-1)
        rtg_batch[:, -1] = rtg
        
        #############################################
        # (3) generate next_obs with predicted retgs
        dec_input = {
            'obs': obs_batch[:, -C:].to(device),
            'act': act_batch[:, -C:].to(device),
            'rew': rew_batch[:, -C:].to(device),
            'rtg': rtg_batch[:, -C:].to(device)
        }
        dec_output = self.model.head.decode(dec_input, dataset_type='trajectory')
        obses = dec_output['obs']
        
        ##############################################
        # (4) generate next sequence and fill next_obs        
        PAD = torch.zeros((N*B,1))
        obs_PAD = torch.zeros((N*B, 1, D))
        
        obs_batch = torch.cat((obs_batch, obs_PAD), 1)
        act_batch = torch.cat((act_batch, PAD), 1).long()
        rew_batch = torch.cat((rew_batch, PAD), 1)
        rtg_batch = torch.cat((rtg_batch, PAD), 1)
        
        # <append> last obs with predicted obs
        obs = obses[:, -1:, :]
        obs = self.model.head.predict(obs)        
        obs_batch[:, -1] = rearrange(obs, 'n 1 d -> n d')

        # reshape to original
        obs_batch = rearrange(obs_batch, '(n b) t d -> n b t d', n=N, b=B)
        act_batch = rearrange(act_batch, '(n b) t -> n b t', n=N, b=B)
        rew_batch = rearrange(rew_batch, '(n b) t -> n b t', n=N, b=B)
        rtg_batch = rearrange(rtg_batch, '(n b) t -> n b t', n=N, b=B)
        
        beams = {
            'obs_batch': obs_batch,
            'act_batch': act_batch,
            'rew_batch': rew_batch,
            'rtg_batch': rtg_batch
        }

        return beams
    
    
    def _select_best_beam(self, beams):
        B = self.beam_width
        C = self.context_len
        N = self.num_envs
        H = self.horizon
        _, _, _, D = beams['obs_batch'].shape
        
        obs_batch = beams['obs_batch'][:, :, -(H+1): -1]
        act_batch = beams['act_batch'][:, :, -(H+1): -1]
        rew_batch = beams['rew_batch'][:, :, -(H+1): -1]
        rtg_batch = beams['rtg_batch'][:, :, -(H+1): -1]

        gammas = torch.pow(self.gamma, torch.arange(H))
        gammas = gammas.repeat(N, B, 1)
        
        G = (torch.sum(gammas * rew_batch, -1) + 
             np.power(self.gamma, H) * rtg_batch[:, :, -1] * self.rtg_scale)
        
        _best_beam_idx = torch.argmax(G, 1)
        best_beam_idx = rearrange(_best_beam_idx, 'n -> n 1 1').repeat(1, 1, H) 
        best_obs_beam_idx = rearrange(_best_beam_idx, 'n -> n 1 1 1').repeat(1, 1, H, D) 
        
        obs_batch = torch.gather(obs_batch, 1, best_obs_beam_idx).squeeze(1)
        act_batch = torch.gather(act_batch, 1, best_beam_idx).squeeze(1)
        rew_batch = torch.gather(rew_batch, 1, best_beam_idx).squeeze(1)
        rtg_batch = torch.gather(rtg_batch, 1, best_beam_idx).squeeze(1)
        
        beam = {
            'obs_batch': obs_batch,
            'act_batch': act_batch,
            'rew_batch': rew_batch,
            'rtg_batch': rtg_batch
        }

        return beam