import numpy as np

from fedot_ind.core.models.nn.network_impl.forecasting_model.common import (
    DEFAULT_FORECASTING_NN_BATCH_SIZE,
    DEFAULT_FORECASTING_NN_DEVICE,
    DEFAULT_FORECASTING_NN_EPOCHS,
    DEFAULT_FORECASTING_NN_LEARNING_RATE,
    normalize_neural_forecasting_params,
    resolve_neural_patch_length,
)


def test_normalize_neural_forecasting_params_adds_runtime_defaults():
    normalized = normalize_neural_forecasting_params({'patch_len': 12})

    assert normalized['patch_len'] == 12
    assert normalized['epochs'] == DEFAULT_FORECASTING_NN_EPOCHS
    assert normalized['batch_size'] == DEFAULT_FORECASTING_NN_BATCH_SIZE
    assert normalized['learning_rate'] == DEFAULT_FORECASTING_NN_LEARNING_RATE
    assert normalized['device'] == DEFAULT_FORECASTING_NN_DEVICE


def test_resolve_neural_patch_length_reuses_requested_value_when_valid():
    series = np.arange(128, dtype=float)

    assert resolve_neural_patch_length(series, forecast_horizon=8, requested_patch_len=20) == 20


def test_resolve_neural_patch_length_clamps_to_valid_history_range():
    series = np.arange(24, dtype=float)

    resolved = resolve_neural_patch_length(series, forecast_horizon=8, requested_patch_len=64)

    assert resolved == 16
