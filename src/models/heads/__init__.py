from .base import BaseHead
from .atc_head import ATCHead
from .byol_head import BYOLHead
from .bc_head import BCHead
from .bcq_head import BCQHead
from .simclr_head import SimCLRHead
from .dt_head import DTHead
from .trajformer_head import TrajFormerHead
from .curl_head import CURLHead
from .mlr_head import MLRHead
from .identity_head import IdentityHead

__all__ = [
    'BaseHead', 
    'BYOLHead', 
    'BCHead', 
    'BCQHead',
    'SimCLRHead', 
    'TrajFormerHead', 
    'CURLHead', 
    'MLRHead', 
    'IdentityHead', 
    'ATCHead'
]