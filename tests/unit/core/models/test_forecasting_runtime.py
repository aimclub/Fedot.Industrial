import numpy as np
import pytest

pytest.importorskip('torch')

from fedot_ind.core.models.ts_forecasting.forecasting_runtime import (
    TensorDevicePolicy,
    build_hankel_trajectory_transform,
    compute_svd_decomposition,
    series_to_forecast_tensor_batch,
    truncate_decomposition_rank,
)


def test_forecast_tensor_batch_keeps_tensor_native_contract():
    series = np.linspace(1.0, 48.0, num=48)

    batch = series_to_forecast_tensor_batch(
        series,
        forecast_horizon=6,
        device_policy=TensorDevicePolicy(device='cpu'),
    )

    assert batch.history.shape == (48, 1)
    assert batch.forecast_horizon == 6
    assert str(batch.device) == 'cpu'


def test_hankel_runtime_builds_supervised_windows_and_latest_state():
    series = np.linspace(1.0, 60.0, num=60)
    batch = series_to_forecast_tensor_batch(series, forecast_horizon=5)

    transform = build_hankel_trajectory_transform(batch, window_size=12, stride=2)

    assert transform.features.ndim == 2
    assert transform.target.ndim == 2
    assert transform.latest_features.shape == (1, 12)
    assert transform.target.shape[1] == 5
    assert transform.metadata['representation'] == 'supervised_hankel'


def test_low_rank_runtime_preserves_selected_rank_and_basis_shape():
    series = np.sin(np.arange(0, 80, dtype=float) / 5.0) + 0.1 * np.arange(80, dtype=float)
    batch = series_to_forecast_tensor_batch(series, forecast_horizon=6)
    transform = build_hankel_trajectory_transform(batch, window_size=14, stride=1)
    decomposition = compute_svd_decomposition(transform.features)
    truncation = truncate_decomposition_rank(decomposition, explained_variance=0.90, min_rank=2)

    assert truncation.selected_rank >= 2
    assert truncation.basis.shape[1] == truncation.selected_rank
    assert truncation.projected_features.shape[1] == truncation.selected_rank
    assert 0.0 < truncation.explained_variance_retained <= 1.0
