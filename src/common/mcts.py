from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F
import copy
import tqdm

class Node():
    def __init__(self, act, rew, obs, prob, parent): 
        self.act = act      # act is the action from parent to node
        self.rew = rew      # rew is the reward from parent to node
        self.obs = obs      # obs is the node's obs
        
        self.prob = prob    # prob is the probability of action from parent to node (puct only)
        self.parent = parent
        self.children = []
        self.G, self.N  = 0, 0

    def expand_node(self, acts, rews, obses, probs):
        for act, rew, obs, prob in zip(acts, rews, obses, probs):
            obs = rearrange(obs, 'd -> 1 1 d')
            child_node = Node(act, rew, obs, prob, self) # new child node
            self.children.append(child_node)

    def update(self, G):
        self.N += 1
        self.G += G
        
    def is_leaf(self):
        return len(self.children)==0

    def has_parent(self):
        return self.parent is not None

    
class MCTS():
    def __init__(self, 
                 model, 
                 device,
                 action_size, 
                 history_len, 
                 num_simulations, 
                 max_depth, 
                 tree_policy, 
                 c_p):
        
        self.model = model
        self.device = device
        self.action_size = action_size
        self.history_len = history_len
        
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        assert tree_policy in ['uct', 'puct']
        self.tree_policy = tree_policy
        self.c_p = c_p # exploration constant
        
        self.root = Node(None, None, None, None, None)
        
        
    @torch.no_grad()
    def rollout(self, state):
        """
        state: (dict)
            obs_list
            act_list
            rew_list
        """
        for _ in range(self.num_simulations):
            node = self.root
            _state = copy.deepcopy(state)
            
            # (1) select until it reaches the leaf node
            while not node.is_leaf():
                node = self._select(node)
                _state['act_list'].append(node.act)
                _state['rew_list'].append(node.rew)
                _state['obs_list'].append(node.obs)
                
            # (2) expand the leaf node
            self._expand(node, _state)
            
            # (3) simulate from the child node
            node = self._select(node)
            _state['act_list'].append(node.act)
            _state['rew_list'].append(node.rew)
            _state['obs_list'].append(node.obs)

            G = self._simulate(node, _state)
            
            # (4) update statistics from the child node
            self._update(node, G)
            
        Ns = [child.N for child in self.root.children]
        Qs = [child.G / (child.N + 1e-6) for child in self.root.children]
        
        act = np.argmax(Ns)
        self._update_root(act)
        
        return act
            
        
    def _update_root(self, act):
        child = self.root.children[act]
        self.root = child
            
            
    def _select(self, node):
        Qs = []
        Ns = []
        probs = []
        for child in node.children:
            Q = child.G / (child.N + 1e-6)
            N = child.N
            prob = child.prob
            Qs.append(Q)
            Ns.append(N)
            probs.append(prob)
        
        Qs = np.array(Qs)
        Ns = np.array(Ns)
        probs = np.array(probs)
        
        if self.tree_policy == 'uct':    
            # https://www.chessprogramming.org/UCT
            Us = self.c_p * np.sqrt(np.log(np.sum(Ns))/ Ns)
        
        elif self.tree_policy == 'puct':
            # Mastering the game of Go with deep neural networks and tree search
            Us = self.c_p * probs * np.sqrt(1 + np.sum(Ns)) / (1 + Ns)
        
        Ucts = Qs + Us
        child_idx = np.argmax(Ucts)
        
        child_node = node.children[child_idx]
        
        return child_node
            
        
    def _expand(self, node, state):
        """
        generate child node (act, rew, obs') for all possible actions.
        """
        t = self.history_len
        device = self.device
        PAD = 0
        
        # list is mutable object
        state = copy.deepcopy(state)
        
        obs_hist = state['obs_list'][-t:]
        act_hist = state['act_list'][-(t-1):] + [PAD]
        rew_hist = state['rew_list'][-(t-1):] + [PAD]
        
        # (1) get next action probability
        obs_hist = torch.cat(obs_hist, 1)
        act_hist = torch.LongTensor(act_hist).unsqueeze(0)
        rew_hist = torch.FloatTensor(rew_hist).unsqueeze(0)
        
        dec_input = {
            'obs': obs_hist.to(device),
            'act': act_hist.to(device),
            'rew': rew_hist.to(device)
        }
        dec_output = self.model.head.decode(dec_input, dataset_type='trajectory')
        probs = F.softmax(dec_output['act'], -1)
        
        # (2) get next reward given state, action
        A = self.action_size
        obs_hist = obs_hist.repeat(A, 1, 1)
        act_hist = act_hist.repeat(A, 1)
        rew_hist = rew_hist.repeat(A, 1)
        
        act_hist[:,-1] = torch.arange(A)
        
        dec_input = {
            'obs': obs_hist.to(device),
            'act': act_hist.to(device),
            'rew': rew_hist.to(device)
        }
        dec_output = self.model.head.decode(dec_input, dataset_type='trajectory')
        rews = dec_output['rew']
        rew = rews[: ,-1, :].squeeze(-1)
        
        # (3) get next reward given state, action
        rew_hist[:,-1] = rew

        dec_input = {
            'obs': obs_hist.to(device),
            'act': act_hist.to(device),
            'rew': rew_hist.to(device)
        }
        dec_output = self.model.head.decode(dec_input, dataset_type='trajectory')
        obses = dec_output['obs']
        
        # allocate to the child node
        acts = np.arange(self.action_size)
        rews = rews[:, -1, :].squeeze(-1).cpu()
        obses = obses[:, -1, :].cpu()
        probs = probs[:, -1, :].squeeze(0).cpu()

        node.expand_node(acts, rews, obses, probs)
        
        
    def _simulate(self, node, state):
        """
        simulate until max_depth
        """
        t = self.history_len
        device = self.device
        PAD = 0
        
        # list is mutable object
        state = copy.deepcopy(state)
        
        G = 0
        cur_depth = 0
        while cur_depth < self.max_depth:
            # initialize 
            obs_hist = state['obs_list'][-t:]
            act_hist = state['act_list'][-(t-1):] + [PAD]
            rew_hist = state['rew_list'][-(t-1):] + [PAD]

            # (1) get next action
            obs_hist = torch.cat(obs_hist, 1)
            act_hist = torch.LongTensor(act_hist).unsqueeze(0)
            rew_hist = torch.FloatTensor(rew_hist).unsqueeze(0)
        
            dec_input = {
                'obs': obs_hist.to(device),
                'act': act_hist.to(device),
                'rew': rew_hist.to(device)
            }
            dec_output = self.model.head.decode(dec_input, dataset_type='trajectory')
            probs = F.softmax(dec_output['act'], -1)[:, -1, :]
            m = torch.distributions.Categorical(probs)
            act = m.sample()
            
            # (2) get next reward
            act_hist[:, -1] = act
            dec_input = {
                'obs': obs_hist.to(device),
                'act': act_hist.to(device),
                'rew': rew_hist.to(device)
            }
            dec_output = self.model.head.decode(dec_input, dataset_type='trajectory')
            rews = dec_output['rew']
            rew = rews[: ,-1, :].squeeze(-1)
            
            # (3) get next obs
            rew_hist[: , -1] = rew
            dec_input = {
                'obs': obs_hist.to(device),
                'act': act_hist.to(device),
                'rew': rew_hist.to(device)
            }
            dec_output = self.model.head.decode(dec_input, dataset_type='trajectory')
            obses = dec_output['obs']
            obs = obses[: ,-1:, :].cpu()
            
            # (4) update state info and proceed
            act, rew = act.item(), rew.item()
            state['obs_list'].append(obs)
            state['act_list'].append(act)
            state['rew_list'].append(rew)
            
            cur_depth += 1
            G += rew
            
        return G
    
    
    def _update(self, node, G):
        while node is not None:
            node.update(G)
            node = node.parent
            