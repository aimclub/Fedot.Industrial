import numpy as np

from fedot_ind.core.models.ts_forecasting.regime_diagnostics import analyze_regime_diagnostics
from fedot_ind.core.models.ts_forecasting.regime_routing import adapter_name_to_family, recommend_forecasting_model


def test_regime_routing_prefers_periodic_models_for_periodic_series():
    time = np.arange(180, dtype=float)
    series = np.sin(2 * np.pi * time / 24.0)

    decision = recommend_forecasting_model(analyze_regime_diagnostics(series))

    assert decision.primary_adapter in {'mssa', 'ssa_compat'}
    assert decision.regime_hint == 'periodic'


def test_regime_routing_prefers_havok_for_switching_series():
    time = np.arange(120, dtype=float)
    series = np.sin(2 * np.pi * time / 10.0)
    series[30:36] += 3.0
    series[60:72] -= 2.5

    decision = recommend_forecasting_model(analyze_regime_diagnostics(series))

    assert decision.primary_adapter == 'havok'
    assert decision.regime_hint == 'switching'


def test_regime_routing_prefers_okhs_for_locally_linear_series():
    time = np.arange(120, dtype=float)
    series = 0.15 * time + 0.2 * np.sin(time / 12.0)

    decision = recommend_forecasting_model(analyze_regime_diagnostics(series))

    assert decision.primary_adapter == 'okhs'
    assert decision.regime_hint == 'locally_linear'


def test_regime_routing_falls_back_for_short_history():
    series = np.array([1.0, 1.5, 1.2, 1.6, 1.4])

    decision = recommend_forecasting_model(analyze_regime_diagnostics(series))

    assert decision.primary_adapter == 'naive_last_value'
    assert decision.fallback_adapter == 'naive_last_value'


def test_adapter_name_to_family_maps_new_composite_models():
    assert adapter_name_to_family('lagged_ridge_forecaster') == 'lagged_linear'
    assert adapter_name_to_family('low_rank_lagged_ridge_forecaster') == 'low_rank_linear'
    assert adapter_name_to_family('hybrid_ensemble_forecaster') == 'operator_model'
