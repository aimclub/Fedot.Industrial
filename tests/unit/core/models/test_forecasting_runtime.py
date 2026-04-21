import numpy as np
import pytest

pytest.importorskip('torch')

from fedot_ind.core.models.ts_forecasting.forecasting_runtime import (
    ForecastingSplitKind,
    ForecastingSplitSpec,
    MLPForecastingHead,
    TensorDevicePolicy,
    build_hankel_trajectory_transform,
    compute_svd_decomposition,
    iter_forecasting_splits,
    series_to_forecast_tensor_batch,
    truncate_decomposition_rank,
)
from fedot_ind.core.models.ts_forecasting.progress_policy import ForecastingProgressPolicy


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


def test_tensor_device_policy_defaults_to_auto_resolution():
    policy = TensorDevicePolicy()

    assert policy.device == 'auto'
    assert str(policy.resolve_device()) in {'cpu', 'cuda:0'}


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


def test_iter_forecasting_splits_supports_time_series_split():
    series = np.linspace(1.0, 72.0, num=72)
    batch = series_to_forecast_tensor_batch(series, forecast_horizon=6)

    folds = iter_forecasting_splits(
        batch,
        ForecastingSplitSpec(
            kind=ForecastingSplitKind.TIME_SERIES_SPLIT,
            validation_horizon=6,
            n_splits=3,
            gap=2,
        ),
    )

    assert len(folds) == 3
    assert folds[-1].train_end == folds[-1].test_start - 2
    assert all(len(fold.validation_target) == 6 for fold in folds)


def test_iter_forecasting_splits_supports_expanding_window():
    series = np.linspace(1.0, 96.0, num=96)
    batch = series_to_forecast_tensor_batch(series, forecast_horizon=8)

    folds = iter_forecasting_splits(
        batch,
        ForecastingSplitSpec(
            kind=ForecastingSplitKind.EXPANDING_WINDOW,
            validation_horizon=8,
            initial_window=32,
            step_length=8,
            n_splits=3,
        ),
    )

    assert len(folds) == 3
    assert folds[0].train_start == 0
    assert all(fold.train_start == 0 for fold in folds)
    assert folds[0].train_end < folds[-1].train_end


def test_iter_forecasting_splits_supports_rolling_window():
    series = np.linspace(1.0, 120.0, num=120)
    batch = series_to_forecast_tensor_batch(series, forecast_horizon=10)

    folds = iter_forecasting_splits(
        batch,
        ForecastingSplitSpec(
            kind=ForecastingSplitKind.ROLLING_WINDOW,
            validation_horizon=10,
            max_train_size=30,
            step_length=10,
            n_splits=4,
        ),
    )

    assert len(folds) == 4
    assert all(fold.train_end - fold.train_start == 30 for fold in folds)
    assert folds[0].train_start < folds[-1].train_start


def test_mlp_forecasting_head_uses_tqdm_progress(monkeypatch):
    calls = []

    def fake_tqdm(iterable=None, *args, **kwargs):
        calls.append(kwargs.get('desc'))
        return iterable

    monkeypatch.setattr(
        'fedot_ind.core.models.ts_forecasting.forecasting_runtime.tqdm',
        fake_tqdm,
    )

    head = MLPForecastingHead(epochs=3, show_progress=True)
    features = np.linspace(1.0, 12.0, num=12).reshape(6, 2)
    target = np.linspace(1.0, 6.0, num=6).reshape(6, 1)

    head.fit(features, target)

    assert calls
    assert any(desc == 'MLP head fit' for desc in calls)


def test_mlp_forecasting_head_uses_progress_policy(monkeypatch):
    calls = []

    def fake_tqdm(iterable=None, *args, **kwargs):
        calls.append(kwargs.get('desc'))
        return iterable

    monkeypatch.setattr(
        'fedot_ind.core.models.ts_forecasting.forecasting_runtime.tqdm',
        fake_tqdm,
    )

    head = MLPForecastingHead(
        epochs=2,
        progress_policy=ForecastingProgressPolicy(enabled=True, head_training_enabled=True),
    )
    features = np.linspace(1.0, 12.0, num=12).reshape(6, 2)
    target = np.linspace(1.0, 6.0, num=6).reshape(6, 1)

    head.fit(features, target)

    assert any(desc == 'MLP head fit' for desc in calls)
    assert head.get_diagnostics()['progress_policy']['head_training_enabled'] is True
