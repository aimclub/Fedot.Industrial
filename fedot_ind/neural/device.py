"""Legacy tensor placement semantics shared by task adapters.

This compatibility policy retains its historical CPU fallback. New explicit
device admission belongs to optimizer.compute, not to this legacy policy.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TensorDevicePolicy:
    device: str = 'auto'
    dtype: str = 'float32'

    def resolve_device(self) -> torch.device:
        requested = str(self.device).lower()
        if requested == 'auto':
            return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if requested.startswith('cuda') and torch.cuda.is_available():
            return torch.device(requested)
        return torch.device('cpu')

    def resolve_dtype(self) -> torch.dtype:
        try:
            return getattr(torch, str(self.dtype))
        except AttributeError:
            return torch.float32
