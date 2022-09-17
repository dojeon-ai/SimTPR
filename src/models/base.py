import torch
import torch.nn as nn


class TrainerModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        """
        [backbone]: (n,t,c,h,w)-> Tuple((n,t,d), info)
        [head]: (n,t,d)-> Tuple((n,t,d), info)
        """
        x, b_info = self.backbone(x)
        x, h_info = self.head(x)
        info = {
            'backbone': b_info,
            'head': h_info
        }
        return x, info


class AgentModel(nn.Module):
    def __init__(self, backbone, neck, policy):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.policy = policy

    def forward(self, x):
        """
        [backbone]: (n,t,c,h,w)-> Tuple((n,t,d), info)
        [neck]:   (n,t,d)-> Tuple((n,d), info)
        [policy]: (n,d)-> Tuple((n,), info)
        """
        x, b_info = self.backbone(x)
        x, n_info = self.neck(x)
        x, p_info = self.policy(x)
        info = {
            'backbone': b_info,
            'neck': n_info,
            'policy': p_info
        }
        return x, info

