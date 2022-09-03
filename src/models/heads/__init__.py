from .base import BaseHead
from .byol_head import BYOLHead
from .simclr_head import SimCLRHead
from .curl_head import CURLHead
from .mlr_head import MLRHead
from .vit_head import VITHead
from .drloc_head import DRLocHead
from .vqvae_head import VQVAEHead
from .recon_head import ReconHead

__all__ = [
    'BaseHead', 'BYOLHead', 'SimCLRHead', 'CURLHead', 'MLRHead', 'VITHead', 'DRLocHead', 'VQVAEHead', 'ReconHead'
]