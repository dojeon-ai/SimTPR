from .base import BaseHead
from .byol_head import BYOLHead
from .simclr_head import SimCLRHead
from .curl_head import CURLHead
from .mlr_head import MLRHead
from .mae_head import MAEHead

__all__ = [
    'BaseHead', 'BYOLHead', 'SimCLRHead', 'CURLHead', 'MLRHead', 'MAEHead'
]