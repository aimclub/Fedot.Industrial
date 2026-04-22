import numpy as np

from fedot_ind.core.models.ts_forecasting.dmd_models.havok_forecaster import HAVOKForecaster


def _switching_series(length: int = 96) -> np.ndarray:
    time = np.arange(length, dtype=float)
    series = np.sin(2.0 * np.pi * time / 12.0)
    series[28:40] += 2.5
    series[56:72] -= 1.8
    series[72:] += np.linspace(0.0, 1.5, num=length - 72)
    return series


def test_havok_forecaster_produces_forecast_and_event_diagnostics():
    series = _switching_series()
    model = HAVOKForecaster(
        forecast_horizon=8,
        window_size=16,
        rank=4,
        forcing_threshold_scale=0.75,
        forcing_decay=0.9,
    )

    model.fit(series)
    forecast = model.predict(series)
    diagnostics = model.get_diagnostics()

    assert forecast.shape == (8,)
    assert diagnostics['selected_rank'] >= 2
    assert diagnostics['state_dimension'] >= 1
    assert diagnostics['model_family'] == 'operator_model'
    assert diagnostics['trajectory_transform']['kind'] == 'hankel'
    assert diagnostics['decomposition']['strategy'] == 'full'
    assert diagnostics['rank_truncation']['selected_rank'] >= 2
    assert diagnostics['forecast_head']['head_type'] == 'havok_head'
    assert diagnostics['forecast_head']['head_policy'] == 'mlp'
    assert len(diagnostics['forcing_values']) > 0
    assert 'forecast_forcing_values' in diagnostics
    assert len(diagnostics['forecast_forcing_mask']) == 8
    assert isinstance(diagnostics['forcing_active_intervals'], list)


def test_havok_forecaster_supports_linear_head_fallback():
    series = _switching_series()
    model = HAVOKForecaster(
        forecast_horizon=6,
        window_size=16,
        rank=4,
        head_policy='linear',
    )

    model.fit(series)
    forecast = model.predict(series)
    diagnostics = model.get_diagnostics()

    assert forecast.shape == (6,)
    assert diagnostics['forecast_head']['head_policy'] == 'linear'


def test_havok_forecaster_uses_mlp_head_activation_and_depth_contract():
    series = _switching_series()
    model = HAVOKForecaster(
        forecast_horizon=6,
        window_size=16,
        rank=4,
        head_policy='mlp',
        head_activation='gelu',
        head_depth=4,
        head_base_hidden_dim=64,
        head_epochs=8,
        head_learning_rate=1e-3,
    )

    model.fit(series)
    diagnostics = model.get_diagnostics()
    head_diagnostics = diagnostics['forecast_head']['state_head_diagnostics']

    assert diagnostics['forecast_head']['head_activation'] == 'gelu'
    assert diagnostics['forecast_head']['head_depth'] == 4
    assert diagnostics['forecast_head']['head_base_hidden_dim'] == 64
    assert head_diagnostics['activation'] == 'gelu'
    assert head_diagnostics['depth'] == 4
    assert head_diagnostics['hidden_dims'] == (64, 32, 16, 8)
