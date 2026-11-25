"""Differentiable Soft-DTW and its separately named centred divergence."""

from math import isfinite
from numbers import Real

import torch


def _validate(left, right, gamma):
    if (not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor)
            or left.ndim != 3 or right.ndim != 3 or not all(left.shape) or not all(right.shape)
            or left.shape[0] != right.shape[0] or left.shape[-1] != right.shape[-1]
            or left.device != right.device or left.dtype != right.dtype
            or left.dtype not in (torch.float32, torch.float64)):
        raise ValueError("Use matching float32/float64 (batch, time, channel) tensors.")
    if isinstance(gamma, bool) or not isinstance(gamma, Real) or not isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive.")


def _soft_dtw(left, right, gamma):
    costs = torch.cdist(left, right, p=2, compute_mode="donot_use_mm_for_euclid_dist").square()
    n, m = costs.shape[1:]
    infinity = costs.new_full((len(costs),), torch.inf)
    previous = [costs.new_zeros((len(costs),))] + [infinity] * m
    for i in range(n):
        current = [infinity]
        for j in range(m):
            neighbors = torch.stack((previous[j], previous[j + 1], current[-1]), dim=0)
            current.append(costs[:, i, j] - gamma * torch.logsumexp(-neighbors / gamma, dim=0))
        previous = current
    return previous[-1]


def _reduce(result, reduction):
    if reduction == "none":
        return result
    if reduction == "mean":
        return result.mean()
    if reduction == "sum":
        return result.sum()
    raise ValueError("Reduction must be none, mean or sum.")


def soft_dtw(left, right, *, gamma=1., reduction="mean"):
    """Return the soft minimum over alignments, not a metric."""
    _validate(left, right, gamma)
    return _reduce(_soft_dtw(left, right, gamma), reduction)


def soft_dtw_divergence(left, right, *, gamma=1., reduction="mean"):
    """Remove the two self-alignment terms (Blondel et al.)."""
    _validate(left, right, gamma)
    value = _soft_dtw(left, right, gamma)
    return _reduce(value - .5 * (_soft_dtw(left, left, gamma) + _soft_dtw(right, right, gamma)), reduction)
