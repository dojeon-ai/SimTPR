from .base import BaseHead
from .atc_head import ATCHead
from .barlow_head import BarlowHead
from .byol_head import BYOLHead
from .bc_head import BCHead
from .bcq_head import BCQHead
from .bert_head import BERTHead
from .idm_head import IDMHead
from .simclr_head import SimCLRHead
from .gpt_head import GPTHead
from .dt_head import DTHead
from .idm_head import IDMHead
from .clt_head import CLTHead
from .curl_head import CURLHead
from .mlr_head import MLRHead
from .identity_head import IdentityHead
from .rssm_head import RSSMHead
from .vae_head import VAEHead

__all__ = [
    'BaseHead',
    'BarlowHead',
    'BYOLHead', 
    'BCHead', 
    'BCQHead',
    'IDMHead', 
    'SimCLRHead', 
    'IDMHead', 
    'CLTHead', 
    'CURLHead', 
    'MLRHead', 
    'IdentityHead', 
    'ATCHead',
    'RSSMHead',
    'VAEHead'
]