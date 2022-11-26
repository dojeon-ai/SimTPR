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
                 action_size, 
                 beam_width,
                 horizon,
                 context_len,
                 gamma):
        
        self.model = model
        self.device = device
        self.action_size = action_size
        
        self.beam_width = beam_width
        self.horizon = horizon
        self.context_len = context_len
        self.gamma = gamma
        
        
    @torch.no_grad()
    def rollout(self, state):
        """
        [param] state
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
        B = self.beam_width
        C = self.context_len
        PAD = 0
        
        obs_list = _state['obs_list'][-C:]
        act_list = _state['act_list'][-(C-1):] + [PAD]
        rew_list = _state['rew_list'][-(C-1):] + [PAD]
        rtg_list = _state['rtg_list'][-(C-1):] + [PAD]
        
        obs_batch = torch.cat(obs_list, 1)
        act_batch = torch.LongTensor(act_list).unsqueeze(0)
        rew_batch = torch.FloatTensor(rew_list).unsqueeze(0)
        rtg_batch = torch.FloatTensor(rtg_list).unsqueeze(0)
        
        # init non-used beams
        obs_batch = obs_batch.repeat(B, 1, 1)
        act_batch = act_batch.repeat(B, 1)
        rew_batch = rew_batch.repeat(B, 1)
        rtg_batch = rtg_batch.repeat(B, 1)
        
        inf = 1e9
        likelihoods = [0.0] + [-inf] * (B-1)
        likelihoods = torch.FloatTensor(likelihoods)
        
        beams = {
            'obs_batch': obs_batch,
            'act_batch': act_batch,
            'rew_batch': rew_batch,
            'rtg_batch': rtg_batch
        }
        return beams, likelihoods
        
        
    def _generate_candidates(self, beams, likelihoods):
        """
        [params] beams: (B, C, x)
        [params] likelihoods: (B)
        
        [returns] candidates:  (B*A, C, x)
        [returns] c_likelihoods: (B*A)
        """
        A = self.action_size
        B = self.beam_width
        C = self.context_len
        device = self.device
        
        obs_batch = beams['obs_batch']
        act_batch = beams['act_batch']
        rew_batch = beams['rew_batch']
        rtg_batch = beams['rtg_batch']
        
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
        obs_batch = obs_batch.repeat(A, 1, 1)
        act_batch = act_batch.repeat(A, 1)
        rew_batch = rew_batch.repeat(A, 1)
        rtg_batch = rtg_batch.repeat(A, 1)
        
        actions = torch.arange(A).repeat(B)
        act_batch[:, -1] = actions
        
        candidates = {
            'obs_batch': obs_batch,
            'act_batch': act_batch,
            'rew_batch': rew_batch,
            'rtg_batch': rtg_batch
        }
        
        probs = torch.log(probs)
        probs = rearrange(probs, 'B A -> (B A)')
        probs = probs.cpu()

        likelihoods = torch.repeat_interleave(likelihoods, A)
        likelihoods += probs
        
        return candidates, likelihoods
    
    
    def _select_candidates(self, candidates, c_likelihoods):
        """
        [params] candidates:  (B*A, C, x)
        [params] c_likelihoods: (B*A)
        
        [returns] beams:  (B, C, x)
        [returns] likelihoods: (B)
        """
        B = self.beam_width
        _, top_B_indices = torch.topk(c_likelihoods, k=B)
        
        obs_batch = torch.index_select(candidates['obs_batch'], 0, top_B_indices)
        act_batch = torch.index_select(candidates['act_batch'], 0, top_B_indices)
        rew_batch = torch.index_select(candidates['rew_batch'], 0, top_B_indices)
        rtg_batch = torch.index_select(candidates['rtg_batch'], 0, top_B_indices)
        
        beams = {
            'obs_batch': obs_batch,
            'act_batch': act_batch,
            'rew_batch': rew_batch,
            'rtg_batch': rtg_batch
        }
        likelihoods = torch.index_select(c_likelihoods, 0, top_B_indices)
        
        return beams, likelihoods
    
        
    def _generate_sequence(self, beams):
        """
        [params] beams: (B, C, x)
        [returns] beams: (B, C, x)
        """
        A = self.action_size
        C = self.context_len
        device = self.device
        B, _, D = beams['obs_batch'].shape
        
        obs_batch = beams['obs_batch']
        act_batch = beams['act_batch']
        rew_batch = beams['rew_batch']
        rtg_batch = beams['rtg_batch']
        
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
        PAD = torch.zeros((B,1))
        obs_PAD = torch.zeros((B, 1, D))
        
        obs_batch = torch.cat((obs_batch, obs_PAD), 1)
        act_batch = torch.cat((act_batch, PAD), 1).long()
        rew_batch = torch.cat((rew_batch, PAD), 1)
        rtg_batch = torch.cat((rtg_batch, PAD), 1)
        
        # <append> last obs with predicted obs
        obs = obses[:, -1:, :]
        obs = self.model.head.predict(obs)
        obs_batch[:, -1] = rearrange(obs, 'n 1 d -> n d')

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
        H = self.horizon
        
        obs_batch = beams['obs_batch'][:, -(H+1): -1]
        act_batch = beams['act_batch'][:, -(H+1): -1]
        rew_batch = beams['rew_batch'][:, -(H+1): -1]
        rtg_batch = beams['rtg_batch'][:, -(H+1): -1]

        gammas = torch.pow(self.gamma, torch.arange(H))
        gammas = gammas.repeat(B, 1)
        G = torch.sum(gammas * rew_batch, 1) + np.power(self.gamma, H) * rtg_batch[:, -1]
        best_beam_idx = torch.argmax(G).item()
                
        beam = {
            'obs_batch': obs_batch[best_beam_idx],
            'act_batch': act_batch[best_beam_idx],
            'rew_batch': rew_batch[best_beam_idx],
            'rtg_batch': rtg_batch[best_beam_idx]
        }

        return beam