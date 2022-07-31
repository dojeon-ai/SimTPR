from .base import BaseHead
from .byol_head import BYOLHead
from .simclr_head import SimCLRHead
from .curl_head import CURLHead


__all__ = [
    'BaseHead', 'BYOLHead', 'SimCLRHead', 'CURLHead'
]