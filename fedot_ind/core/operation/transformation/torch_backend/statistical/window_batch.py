"""Bounded CPU windows with the original per-window reduction geometry.

Quantile ranks and in-place interpolation follow PyTorch 2.6 Sorting.cpp.
Only sorting is shared; enlarging mean/std/lerp batches changes rounding.
"""
from functools import lru_cache

import torch


def _statistics_kernel(x: torch.Tensor, width: int, step: int):
    windows = x.unfold(-1, width, step)
    ordered = torch.sort(windows, dim=-1).values
    count, channels = windows.shape[1], x.shape[0]
    output = torch.empty((channels, count * 9), dtype=x.dtype, device=x.device)
    for i in range(count):
        sample = x[:, i * step:i * step + width]
        output[:, i * 9] = sample.mean(-1)
        if width % 2:
            output[:, i * 9 + 1] = ordered[:, i, width // 2]
        else:
            output[:, i * 9 + 1] = (ordered[:, i, width // 2 - 1] + ordered[:, i, width // 2]) / 2
        output[:, i * 9 + 2] = sample.std(-1, unbiased=False)
        output[:, i * 9 + 3] = sample.max(-1).values
        output[:, i * 9 + 4] = sample.min(-1).values
        for j, q in enumerate([.05, .25, .75, .95]):
            rank = torch.tensor(q, dtype=x.dtype, device=x.device) * (width - 1)
            below = rank.to(torch.long).expand(channels, 1)
            above = rank.ceil().to(torch.long).expand(channels, 1)
            weight = rank - below
            lower = ordered[:, i].gather(-1, below)
            upper = ordered[:, i].gather(-1, above)
            lower.lerp_(upper, weight)
            output[:, i * 9 + 5 + j] = lower[:, 0]
    return output


@lru_cache(maxsize=1)
def _compiled_kernel():
    # Compilation is process-local, lazy, and charged to the first transform.
    return torch.jit.script(_statistics_kernel)


def _reference_windows(values, plan):
    from fedot_ind.core.operation.transformation.representation.statistical.methods import STAT_METHODS_TORCH
    return torch.cat([torch.stack([torch.as_tensor(method(values[:, start:start + plan.width], -1)).reshape(
        values.shape[0]) for method in STAT_METHODS_TORCH.values()], dim=-1) for start in plan.starts], dim=-1)


def window_statistics(values, plan, max_elements):
    if (values.device.type != 'cpu' or values.dtype != torch.float32 or values.ndim != 2
            or values.requires_grad or type(max_elements) is not int or max_elements < 1):
        raise ValueError('Window kernel requires a bounded CPU float32 matrix without gradients')
    channels = values.shape[0]
    if channels < 1 or channels * plan.width > max_elements:
        raise ValueError('One window exceeds the temporary element budget')
    # kthvalue and sort can choose different signs among equal +/-0 values.
    if bool(((values == 0) & torch.signbit(values)).any()):
        return _reference_windows(values, plan)
    step = plan.starts[1] - plan.starts[0] if len(plan.starts) > 1 else plan.width
    windows_per_block = max(1, max_elements // (channels * plan.width))
    parts = []
    kernel = _compiled_kernel()
    for first in range(0, len(plan.starts), windows_per_block):
        starts = plan.starts[first:first + windows_per_block]
        block = values[:, starts[0]:starts[-1] + plan.width]
        parts.append(kernel(block, plan.width, step))
    return torch.cat(parts, dim=-1)
