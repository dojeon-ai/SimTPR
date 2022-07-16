import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, backbone, head, policy):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.policy = policy

    def extract_feat(self, x):
        x = self.backbone(x)
        return x

    def forward(self, x):
        x = self.backbone(x)
        x = self.policy(x)
        return x

    def load_backbone_and_policy(self, checkpoint):
        state_dict = self.state_dict()
        _pretrained_state_dict = checkpoint['model_state_dict']
        pretrained_state_dict = {}
        for name, param in _pretrained_state_dict.items():
            if 'backbone' in name:
                pretrained_state_dict[name] = param
            if name == 'head.projector.0.weight':
                pretrained_state_dict['policy.fc_v.0.weight_mu'] = param
                pretrained_state_dict['policy.fc_adv.0.weight_mu'] = param
                
        self.load_state_dict(pretrained_state_dict, strict=False)
