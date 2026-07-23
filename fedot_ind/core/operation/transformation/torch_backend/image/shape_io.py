from __future__ import annotations

from typing import Any

import torch

from fedot_ind.core.operation.transformation.torch_backend.io import resolve_torch_device


def check_input_shape(X: Any) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Convert supported time-series layouts to a 2D working tensor."""

    if not isinstance(X, torch.Tensor):
        try:
            X = torch.as_tensor(X)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"X must be a torch.Tensor or convertible to one, got {type(X)}"
            ) from exc

    init_shape = tuple(X.shape)
    if X.ndim == 1:
        X = X.unsqueeze(0)
    elif X.ndim == 3:
        batch, channels, timestamps = X.shape
        X = X.reshape(batch * channels, timestamps)
    elif X.ndim > 3:
        raise ValueError(f"X must be 1D, 2D or 3D, got shape={tuple(X.shape)}")

    if not torch.is_floating_point(X):
        X = X.float()

    return X, init_shape


def prepare_series_input(
    X: Any,
    *,
    torch_device: Any = "auto",
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Convert input to float32 tensor on the resolved device."""

    resolved = resolve_torch_device(torch_device)
    X, init_shape = check_input_shape(X)
    return X.to(device=resolved, dtype=torch.float32), init_shape


def convert_to_init_dim(
    X: torch.Tensor,
    init_shape: tuple[int, ...],
) -> torch.Tensor:
    """Restore batch/channel axes after ``check_input_shape`` flattening."""

    if len(init_shape) == 3:
        batch, n_channels = init_shape[0], init_shape[1]
        expected = batch * n_channels
        if X.shape[0] != expected:
            raise ValueError(
                f"Batch/channel flatten mismatch: input shape {init_shape} "
                f"implies {expected} flat samples, got {X.shape[0]}."
            )
        return X.reshape(batch, n_channels, *X.shape[1:])
    return X
