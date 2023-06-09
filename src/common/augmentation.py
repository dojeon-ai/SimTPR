import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as aug


#####################
## ATARI 
class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


class Augmentation(nn.Module):
    def __init__(self, obs_shape, aug_types=[]):
        super().__init__()
        self.layers = []
        for aug_type in aug_types:
            if aug_type == 'random_shift':
                _, _, W, H = obs_shape
                self.layers.append(nn.ReplicationPad2d(4))
                self.layers.append(aug.RandomCrop((W, H)))
            
            elif aug_type == 'cutout':
                self.layers.append(aug.RandomErasing(p=0.5))
            
            elif aug_type == 'h_flip':
                self.layers.append(aug.RandomHorizontalFlip(p=0.1))

            elif aug_type == 'v_flip':
                self.layers.append(aug.RandomVerticalFlip(p=0.1))

            elif aug_type == 'rotate':
                self.layers.append(aug.RandomRotation(degrees=5.0))

            elif aug_type == 'intensity':
                self.layers.append(Intensity(scale=0.05))

            else:
                raise ValueError

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


#####################
## DMC 
class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

