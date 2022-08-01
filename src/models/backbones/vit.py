import torch.nn as nn
import torch
from .base import BaseBackbone
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from src.common.train_utils import orthogonal_init, xavier_uniform_init


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 2, dropout = 0.):
        super().__init__()
        head_dim = dim // heads
        self.heads = heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) 

    def forward(self, x, attn_mask=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'n t (h d) -> n h t d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        # TODO: attn_mask->casual
        out = rearrange(out, 'n h t d -> n t (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, attn_mask=None):
        for attn, ff in self.layers:
            x = attn(x, attn_mask) + x
            x = ff(x) + x
        return x


class VIT(BaseBackbone):
    name = 'vit'
    def __init__(self,
                 obs_shape,
                 action_size,
                 patch_size,
                 t_step,
                 pool,
                 enc_depth,
                 enc_dim, 
                 enc_mlp_dim,
                 enc_heads, 
                 dec_depth, 
                 dec_dim, 
                 dec_mlp_dim,
                 dec_heads, 
                 emb_dropout,
                 dropout, 
                 init_type):

        super().__init__()
        frame, channel, image_height, image_width = obs_shape
        image_channel = frame * channel
        patch_height, patch_width = patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = image_channel * patch_height * patch_width

        assert pool in {'mean'}, 'currently, pool must be mean (mean pooling)'

        ###########################################
        # Encoder 
        self.patch_embedding = nn.Sequential(
            Rearrange('n t c (h p1) (w p2) -> n (t h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, enc_dim),
        )

        self.enc_spatial_embedding = nn.Parameter(torch.randn(1, num_patches + 1, enc_dim)) # +1 for action
        self.enc_temporal_embedding = nn.Parameter(torch.randn(1, t_step, enc_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.encoder = Transformer(dim=enc_dim, 
                                   depth=enc_depth, 
                                   heads=enc_heads, 
                                   mlp_dim=enc_mlp_dim, 
                                   dropout=dropout)

        ########################################
        # Decoder
        self.decoder_embedding = nn.Linear(enc_dim, dec_dim)
        self.act_embedding = nn.Linear(action_size, dec_dim)

        self.dec_spatial_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dec_dim)) # +1 for action
        self.dec_temporal_embedding = nn.Parameter(torch.randn(1, t_step, dec_dim))

        self.decoder = Transformer(dim=dec_dim, 
                                   depth=dec_depth, 
                                   heads=dec_heads, 
                                   mlp_dim=dec_mlp_dim, 
                                   dropout=dropout)

        self._output_dim = dec_dim

    def forward(self, x):
        """
        img: (N, T, C, H, W)
        act: (N, T) 
        done: (N, T)
        img_mask: (N, T)
        """
        img = x['img']
        act = x['act']
        done = x['done']
        img_mask = x['img_mask']


        import pdb
        pdb.set_trace()
        
        x = self.patch_embedding(img)
        # act = self.to_act_embedding(act)
        # x += positional_embedding : spaital + temporal embedding
        # x = self.dropout(x)

        # Transformer Encoder
        # enc_x = torch.select_index(img, img_mask)
        # enc_x = self.encoder(enc_x)
        
        # Transformer Decoder
        # x = mask * x + (1-mask) * enc_x
        # x = cat(x, act)
        # x += positional_embedding : decoder position
        # casual attention mask
        # attn_mask = (item_ids > 0).unsqueeze(1).repeat(1, item_ids.size(1), 1).unsqueeze(1)
        # attn_mask.tril_()
        # x = self.decoder(x)        
        
        return x
