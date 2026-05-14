from __future__ import annotations

from typing import Any

import numpy as np
import torch

from fedot_ind.core.models.ts_forecasting.forecasting_runtime import (
    TensorDevicePolicy,
    resolve_window_size,
)
from fedot_ind.core.operation.transformation.data.trajectory_embedding import estimate_window

DEFAULT_FORECASTING_NN_EPOCHS = 150
DEFAULT_FORECASTING_NN_BATCH_SIZE = 16
DEFAULT_FORECASTING_NN_LEARNING_RATE = 1e-3
DEFAULT_FORECASTING_NN_DEVICE = 'cuda'


def normalize_neural_forecasting_params(params: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = {
        key: value
        for key, value in dict(params or {}).items()
        if value is not None
    }
    normalized.setdefault('epochs', DEFAULT_FORECASTING_NN_EPOCHS)
    normalized.setdefault('batch_size', DEFAULT_FORECASTING_NN_BATCH_SIZE)
    normalized.setdefault('learning_rate', DEFAULT_FORECASTING_NN_LEARNING_RATE)
    normalized.setdefault('device', DEFAULT_FORECASTING_NN_DEVICE)
    return normalized


def resolve_neural_forecasting_device(device: str | None = None) -> torch.device:
    requested_device = str(device or DEFAULT_FORECASTING_NN_DEVICE)
    return TensorDevicePolicy(device=requested_device).resolve_device()


def resolve_neural_patch_length(
        time_series: np.ndarray,
        forecast_horizon: int,
        *,
        requested_patch_len: int | None = None,
        multiplier: float = 1.0,
) -> int:
    values = np.asarray(time_series, dtype=float).reshape(-1)
    resolved_horizon = int(max(1, forecast_horizon))
    max_patch_len = max(2, len(values) - resolved_horizon)
    if requested_patch_len is not None:
        return int(max(2, min(int(requested_patch_len), max_patch_len)))

    estimated_window = int(estimate_window(len(values), forecast_horizon=resolved_horizon))
    resolved_window = resolve_window_size(
        series_length=len(values),
        forecast_horizon=resolved_horizon,
        window_size=estimated_window,
    )
    scaled_window = int(round(float(multiplier) * resolved_window))
    return int(max(2, min(scaled_window, max_patch_len)))


def build_plateau_scheduler(
        optimizer: torch.optim.Optimizer,
        *,
        factor: float = 0.5,
        patience: int = 8,
        min_lr: float = 1e-5,
):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=float(factor),
        patience=int(max(1, patience)),
        min_lr=float(min_lr),
    )
