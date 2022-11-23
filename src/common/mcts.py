import numpy as np
import copy

class Node():
    def __init__(self, act, rew, obs, prob, parent): 
        self.act = act      # act is the action from parent to node
        self.rew = rew      # rew is the reward from parent to node
        self.obs = obs      # obs is the node's obs
        
        self.prob = prob    # prob is the probability of action from parent to node (puct only)
        self.parent = parent
        self.children = []
        self.R, self.N  = 0, 0

    def expand_node(self, actions, probs):
        for act, prob in zip(actions, probs):
            child_node = Node(act, prob, self) # new child node
            self.children.append(child_node)

    def update(self, reward):
        self.N += 1
        self.R += reward
        
    def is_leaf(self):
        return len(self.children)==0

    def has_parent(self):
        return self.parent is not None

    
class MCTS():
    def __init__(self, model, history_len, num_simulations, max_depth, tree_policy, c_p):
        self.model = model
        self.history_len = history_len
        
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        assert tree_policy in ['uct', 'puct']
        self.tree_policy = tree_policy
        self.c_p = c_p # exploration constant
        
        self.root = Node(None, None, None, None, None)
        
        
    def rollout(self, state):
        """
        state: (dict)
            obs_list
            act_list
            rew_list
        """
        ns = 0
        while ns < self.num_simulations:
            node = self.root
            _state = copy.deepcopy(state)
            
            
            # select until it reaches the leaf node
            while not node.is_leaf():
                node = self.select(node)
                
                # update 
                _state['act_list'].append(node.act)
                _state['rew_list'].append(node.rew)
                
            # expand the leaf node
            self.expand(_state)

            
            
            
            
    def select(self, node):
        Qs = []
        Ns = []
        probs = []
        for child in node.children:
            Q = child.R / (child.N + 1e-6)
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
            Us = self.c_p * probs * np.sqrt(np.sum(Ns)) / (1 + Ns)
        
        Ucts = Qs + Us
        child_idx = np.argmax(Ucts)
        
        child_node = node.children[child_idx]
        
        return child_node
            
    def expand(self, state):
        t = self.history_len
        
        import pdb
        pdb.set_trace()
        
        obs_hist = torch.cat(state['obs_list'][-t:], 1)
        act_hist = torch.LongTensor(state['act_list'][-t:]).unsqueeze(0)
        rew_hist = torch.FloatTensor(state['rew_list'][-t:]).unsqueeze(0)
        
        
        
        
            
    
"""
def mcts(state):
    root_node  = Node(None, None)
    while time remains:
        n, s = root_node, copy.deepcopy(state)
        while not n.is_leaf():    # select leaf
          n = tree_policy_child(n)
          s.addmove(n.move)
        n.expand_node(s)          # expand
        n = tree_policy_child(n)
        while not terminal(s):    # simulate
          s = simulation_policy_child(s)
        result = evaluate(s)
        while n.has_parent():     # propagate
          n.update(result)
          n = n.parent

return best_move(tree)
"""