import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, backbone, header, policy):
        super().__init__()
        self.backbone = backbone
        self.header = header
        self.policy = policy

    def extract_feat(self, x):
        x = self.backbone(x)
        return x

    def forward(self, x):
        x = self.backbone(x)
        x = self.policy(x)
        return x
