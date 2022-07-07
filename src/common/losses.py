import torch.nn as nn
import torch


class TemporalContrastiveLoss(nn.Module):
    # Temporal extension of SimCLR loss
    # Identical to the SimCLR loss if T=1

    def __init__(self):
        super().__init__()

    def forward(self, z1, z2):

        return loss


class TemporalConsistencyLoss(nn.Module):
    # Temporal extension of BYOL loss
    # Identical to the BYOL loss if T=1

    def __init__(self):
        super().__init__()

    def forward(self, p, z):

        return loss

    



if __name__ == '__main__':
    print('[TEST LOSS Functions]')

