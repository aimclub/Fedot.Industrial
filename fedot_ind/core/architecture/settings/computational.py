import torch
from fastcore.basics import defaults


def _has_mps():
    "Check if MPS is available - modified from fastai"
    return False
    # return nested_attr(torch, "backends.mps.is_available", noop)()


def default_device(device_type: str = 'CUDA'):
    "Return or set default device; `use_cuda`: -1 - CUDA/mps if available; True - error if not available; False - CPU"
    if device_type == 'CUDA':
        device_type = defaults.use_cuda
    else:
        defaults.use_cuda = False
    if device_type is None:
        if torch.cuda.is_available() or _has_mps():
            device_type = True
    if device_type:
        if torch.cuda.is_available():
            return torch.device(torch.cuda.current_device())
        if _has_mps():
            return torch.device("mps")
    return torch.device("cpu")
