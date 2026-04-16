from __future__ import annotations

from .neural_forecast_head import (
    NEURAL_FORECASTING_MODEL_REGISTRY,
    NeuralForecastHead,
    NeuralForecastHeadSpec,
    build_neural_forecast_head,
    build_neural_forecasting_input_data,
    build_neural_forecasting_stage_diagnostics,
    normalize_neural_forecast_prediction,
    resolve_neural_forecasting_model_cls,
)


class NeuralForecastHeadBridge(NeuralForecastHead):
    """
    Compatibility wrapper kept for Phase 2/3 transition.

    The primitive-oriented source-of-truth now lives in
    `neural_forecast_head.py`, while this class preserves the previous
    public entrypoint for benchmark/runtime code that still imports the
    bridge by name.
    """

    def __init__(self, model_name: str, forecast_horizon: int, params: dict | None = None):
        super().__init__(
            spec=NeuralForecastHeadSpec(
                model_name=model_name,
                forecast_horizon=forecast_horizon,
                params=params,
            )
        )


__all__ = [
    'NEURAL_FORECASTING_MODEL_REGISTRY',
    'NeuralForecastHead',
    'NeuralForecastHeadBridge',
    'NeuralForecastHeadSpec',
    'build_neural_forecast_head',
    'build_neural_forecasting_input_data',
    'build_neural_forecasting_stage_diagnostics',
    'normalize_neural_forecast_prediction',
    'resolve_neural_forecasting_model_cls',
]
