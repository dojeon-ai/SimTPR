from .base import BaseHead
from .atc_head import ATCHead
from .byol_head import BYOLHead
from .bc_head import BCHead
from .simclr_head import SimCLRHead
from .trajformer_head import TrajFormerHead
from .curl_head import CURLHead
from .mlr_head import MLRHead
from .identity_head import IdentityHead

__all__ = [
    'BaseHead', 
    'BYOLHead', 
    'BCHead', 
    'SimCLRHead', 
    'TrajFormerHead', 
    'CURLHead', 
    'MLRHead', 
    'IdentityHead', 
    'ATCHead'
]