"""Collect layerwise topology on a fixed, caller-owned evaluation sample."""

import numpy as np
import torch

from .window_analysis import _summary, persistence


def activation_topology(model, inputs, layers, *, sample_size=128, seed=0):
    if type(sample_size) is not int or sample_size < 2 or len(inputs) < 2:
        raise ValueError("At least two inputs and two sample slots are required.")
    modules = dict(model.named_modules())
    names = tuple(layers)
    if not names or len(set(names)) != len(names) or any(name not in modules for name in names):
        raise ValueError("Select unique existing layers.")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(inputs), min(len(inputs), sample_size), replace=False))
    original_modes = {module: module.training for module in modules.values()}
    captured, hooks = {}, []

    def collect(name):
        def hook(_module, _args, result):
            if name in captured or not isinstance(result, torch.Tensor) or result.ndim < 2:
                raise ValueError("A selected layer must emit one batched tensor per forward.")
            value = result if result.ndim == 2 else result.flatten(start_dim=2).mean(dim=2)
            if value.dtype not in (torch.float32, torch.float64):
                value = value.float()
            captured[name] = value.detach().cpu().numpy()
        return hook

    try:
        hooks = [modules[name].register_forward_hook(collect(name)) for name in names]
        model.eval()
        with torch.inference_mode():
            model(inputs[indices])
    finally:
        for hook in hooks:
            hook.remove()
        for module, mode in original_modes.items():
            module.training = mode
    if set(captured) != set(names):
        raise ValueError("At least one selected layer was not called by the model.")
    return {name: {"persistence": _summary(persistence(cloud, max_points=sample_size)),
                   "rank": int(np.linalg.matrix_rank(cloud - cloud.mean(axis=0)))}
            for name, cloud in captured.items()}
