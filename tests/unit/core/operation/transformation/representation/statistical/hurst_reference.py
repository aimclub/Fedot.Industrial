"""Pre-optimization implementations, kept independent as numerical oracles."""

import numpy as np
import torch


def numpy_hurst(array, axis=None):
    if axis == 2:
        return None
    x = np.array(array)
    n = x.size
    t = np.arange(1, n + 1)
    y = np.cumsum(x)
    average = y / t
    deviations = np.zeros(n)
    ranges = np.zeros(n)
    for i in range(n):
        deviations[i] = np.std(x[:i + 1])
        centered = y - t * average[i]
        ranges[i] = np.ptp(centered[:i + 1])
    with np.errstate(divide='ignore', invalid='ignore'):
        response = np.log(ranges / deviations)[1:]
    design = np.column_stack((np.log(t)[1:], np.ones(n - 1)))
    return np.linalg.lstsq(design, response, rcond=None)[0][0]


def torch_hurst(values, axis=-1):
    if (axis != -1) | (axis != len(values.shape)):
        x = values.transpose(axis, -1)
    if values.ndim > 2:
        x = values.reshape(-1, values.shape[-1])
    if values.ndim == 1:
        x = values.unsqueeze(0)
    batch, length = x.shape
    t = torch.arange(1, length + 1, device=x.device, dtype=x.dtype).unsqueeze(0)
    y = torch.cumsum(x, dim=-1)
    average = y / t
    deviations = torch.zeros_like(x)
    ranges = torch.zeros_like(x)
    for i in range(length):
        deviations[:, i] = x[:, :i + 1].std(dim=-1, unbiased=False)
        centered = y - t * average[:, i].unsqueeze(1)
        ranges[:, i] = centered[:, :i + 1].amax(dim=-1) - centered[:, :i + 1].amin(dim=-1)
    response = torch.log((ranges / (deviations + 1e-12))[:, 1:])
    n = torch.log(t[:, 1:]).squeeze(0)
    design = torch.stack((n, torch.ones_like(n)), dim=1)
    result = torch.empty(batch, device=x.device, dtype=x.dtype)
    for b in range(batch):
        result[b] = torch.linalg.lstsq(design, response[b].unsqueeze(1)).solution.squeeze()[0]
    if values.ndim > 2:
        result = result.reshape(values.shape[0], values.shape[1])
    return result if result.numel() > 1 else result.item()
