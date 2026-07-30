"""Shared tensor IO helpers for torch-backed transformations."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_numpy(values: Any) -> np.ndarray:
    """Convert common tensor/dataframe-like values to a NumPy array."""

    if hasattr(values, "detach") and callable(values.detach):
        values = values.detach().cpu().numpy()
    elif (
        hasattr(values, "cpu")
        and callable(values.cpu)
        and values.__class__.__module__.startswith("torch")
    ):
        values = values.cpu().numpy()
    elif hasattr(values, "values") and not isinstance(values, np.ndarray):
        values = values.values
    return np.asarray(values)


def normalize_time_series_tensor(values: Any) -> np.ndarray:
    """Normalize time-series input to ``(n_samples, n_channels, n_timestamps)``."""

    array = to_numpy(values).astype(float, copy=False)
    if array.ndim == 0:
        raise ValueError("X must contain at least one sample.")
    if array.ndim == 1:
        return array.reshape(1, 1, -1)
    if array.ndim == 2:
        return array.reshape(array.shape[0], 1, array.shape[1])
    if array.ndim == 3:
        return array
    return array.reshape(array.shape[0], int(np.prod(array.shape[1:-1])), array.shape[-1])


def resolve_torch_device(device: Any = "auto"):
    """Resolve ``auto``/string/device values to a concrete ``torch.device``."""

    import torch

    if isinstance(device, torch.device):
        resolved = device
    else:
        requested = "auto" if device is None else str(device).strip().lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "torch_device='cuda' was requested, but CUDA is not available."
        )
    return resolved


def to_torch(values: Any, *, device: Any = "auto"):
    """Convert values to ``torch.float32`` on a resolved device."""

    import torch

    resolved_device = resolve_torch_device(device)
    if isinstance(values, torch.Tensor):
        return values.to(device=resolved_device, dtype=torch.float32)
    return torch.as_tensor(to_numpy(values), dtype=torch.float32, device=resolved_device)
