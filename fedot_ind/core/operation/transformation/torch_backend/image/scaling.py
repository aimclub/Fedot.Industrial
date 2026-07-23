from __future__ import annotations

import torch


def per_sample_minmax_scale(
    X: torch.Tensor,
    *,
    feature_range: tuple[float, float] = (-1.0, 1.0),
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Scale each sample independently to the requested feature range."""

    min_value = X.amin(dim=dim, keepdim=True)
    max_value = X.amax(dim=dim, keepdim=True)
    scale = (max_value - min_value).clamp_min(eps)

    low, high = feature_range
    X_scaled = (X - min_value) / scale
    X_scaled = X_scaled * (high - low) + low
    return torch.nan_to_num(X_scaled)
