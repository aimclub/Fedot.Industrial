"""Frozen pre-EVO-07 tensor kernels, independent of optimized dispatch."""
import torch

from fedot_ind.core.operation.transformation.torch_backend.image.paa import segmentation_torch


def digitize_reference(x, bins):
    return (bins[:, None, :] < x[:, :, None]).sum(-1).long()


def gasf_reference(cosine, sine):
    return cosine.unsqueeze(2) * cosine.unsqueeze(1) - sine.unsqueeze(2) * sine.unsqueeze(1)


def mtf_reference(binned, transition, image_size):
    batch, length = binned.shape
    rows = torch.arange(batch, device=binned.device)[:, None, None]
    field = transition[rows, binned[:, :, None], binned[:, None, :]]
    window, remainder = divmod(length, image_size)
    if not remainder:
        return field.reshape(batch, image_size, window, image_size, window).mean((2, 4))
    start, end, _ = segmentation_torch(length, window + 1, False, image_size, device=binned.device)
    integral = torch.nn.functional.pad(field.cumsum(1).cumsum(2), (1, 0, 1, 0))
    r0, r1 = start[:, None], end[:, None]
    c0, c1 = start[None, :], end[None, :]
    total = (integral[:, r1, c1] - integral[:, r0, c1]
             - integral[:, r1, c0] + integral[:, r0, c0])
    return total / ((r1 - r0) * (c1 - c0)).to(transition.dtype).clamp_min(1)
