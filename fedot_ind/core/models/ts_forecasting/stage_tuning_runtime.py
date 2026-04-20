from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from fedot_ind.core.models.ts_forecasting.forecasting_runtime import (
    ForecastingEvaluationResult,
    ForecastingSplitSpec,
    TensorDevicePolicy,
    evaluate_forecast,
    series_to_forecast_tensor_batch,
    split_forecasting_batch,
)
from fedot_ind.core.models.ts_forecasting.stage_tuning_execution import (
    ForecastingSequentialStageTuningResult,
    run_sequential_stage_tuning,
)
from fedot_ind.core.repository.forecasting_registry import canonical_forecasting_model_name

try:
    from .neural_forecast_head import build_neural_forecast_head
except Exception:  # pragma: no cover - lightweight envs may miss neural runtime deps
    build_neural_forecast_head = None

try:
    from .okhs_fdmd_forecaster import build_okhs_fdmd_forecaster
except Exception:  # pragma: no cover - lightweight envs may miss operator runtime deps
    build_okhs_fdmd_forecaster = None


@dataclass(frozen=True)
class ForecastingSeriesEvaluation:
    model_name: str
    canonical_model_name: str
    family: str
    parameters: dict[str, Any]
    forecast_horizon: int
    metric: ForecastingEvaluationResult
    forecast: tuple[float, ...]
    target: tuple[float, ...]
    split_metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'model_name': self.model_name,
            'canonical_model_name': self.canonical_model_name,
            'family': self.family,
            'parameters': dict(self.parameters),
            'forecast_horizon': int(self.forecast_horizon),
            'metric': self.metric.to_dict(),
            'forecast': list(self.forecast),
            'target': list(self.target),
            'split_metadata': dict(self.split_metadata),
            'diagnostics': dict(self.diagnostics),
            **self.metadata,
        }


@dataclass(frozen=True)
class ForecastingSeriesStageTuningResult:
    model_name: str
    canonical_model_name: str
    family: str
    sequential_result: ForecastingSequentialStageTuningResult
    baseline_evaluation: ForecastingSeriesEvaluation
    best_evaluation: ForecastingSeriesEvaluation
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'model_name': self.model_name,
            'canonical_model_name': self.canonical_model_name,
            'family': self.family,
            'sequential_result': self.sequential_result.to_dict(),
            'baseline_evaluation': self.baseline_evaluation.to_dict(),
            'best_evaluation': self.best_evaluation.to_dict(),
            **self.metadata,
        }


def _normalize_base_params(params: dict[str, Any] | None, *, model_name: str | None = None) -> dict[str, Any]:
    normalized = {
        key: value
        for key, value in dict(params or {}).items()
        if value is not None
    }
    canonical_name = canonical_forecasting_model_name(model_name) if model_name else ''
    if canonical_name == 'lagged_forecaster':
        normalized.setdefault('channel_model', 'ridge')
        normalized.setdefault('window_size', 10)
        normalized.setdefault('stride', 1)
        normalized.setdefault('alpha', 1.0)
    elif canonical_name in {'mssa_forecaster', 'ssa_forecaster'}:
        normalized.setdefault('head_policy', 'mlp')
        normalized.setdefault('head_hidden_dim', 64)
        normalized.setdefault('head_hidden_layers', 2)
        normalized.setdefault('head_epochs', 120)
        normalized.setdefault('head_learning_rate', 1e-3)
        normalized.setdefault('device', 'cpu')
    elif canonical_name == 'havok_forecaster':
        normalized.setdefault('head_policy', 'mlp')
        normalized.setdefault('head_hidden_dim', 64)
        normalized.setdefault('head_hidden_layers', 2)
        normalized.setdefault('head_epochs', 120)
        normalized.setdefault('head_learning_rate', 1e-3)
        normalized.setdefault('device', 'cpu')
    return normalized


def _squeeze_series_if_univariate(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and array.shape[1] == 1:
        return array.reshape(-1)
    return array


def _prepare_runtime_series(
        canonical_model_name: str,
        series: np.ndarray,
        params: dict[str, Any],
) -> np.ndarray:
    prepared = np.asarray(series, dtype=float)
    if canonical_model_name == 'ssa_forecaster':
        history_lookback = int(params.get('history_lookback', 0) or 0)
        if history_lookback > 0 and prepared.shape[0] > history_lookback:
            prepared = prepared[-history_lookback:]
    return _squeeze_series_if_univariate(prepared)


def _seasonal_naive_forecast(train: np.ndarray, horizon: int, seasonal_period: int) -> np.ndarray:
    lag = seasonal_period if seasonal_period > 1 and len(train) > seasonal_period else 1
    base = train[-lag:]
    repeats = int(np.ceil(horizon / lag))
    return np.tile(base, repeats)[:horizon]


def _mase_scale(train: np.ndarray, seasonal_period: int) -> float:
    lag = seasonal_period if seasonal_period > 1 and len(train) > seasonal_period else 1
    if len(train) <= lag:
        return 1.0
    scale = float(np.mean(np.abs(train[lag:] - train[:-lag])))
    return scale if scale > 1e-8 else 1.0


def _evaluate_forecast_metric(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        y_train: np.ndarray,
        metric_name: str,
        seasonal_period: int,
) -> ForecastingEvaluationResult:
    metric = str(metric_name).lower()
    if metric in {'rmse', 'mae'}:
        return evaluate_forecast(y_true, y_pred, metric_name=metric)

    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    train = np.asarray(y_train, dtype=float).reshape(-1)
    if len(actual) != len(predicted):
        raise ValueError('y_true and y_pred must have the same length.')

    if metric == 'smape':
        denominator = np.abs(actual) + np.abs(predicted)
        denominator = np.where(denominator == 0, 1e-8, denominator)
        per_horizon = 100.0 * 2.0 * np.abs(predicted - actual) / denominator
        metric_value = float(np.mean(per_horizon))
    elif metric == 'mase':
        per_horizon = np.abs(actual - predicted) / _mase_scale(train, seasonal_period)
        metric_value = float(np.mean(per_horizon))
    elif metric == 'owa':
        baseline = _seasonal_naive_forecast(train, len(actual), seasonal_period)
        smape_eval = _evaluate_forecast_metric(
            actual,
            predicted,
            y_train=train,
            metric_name='smape',
            seasonal_period=seasonal_period,
        )
        mase_eval = _evaluate_forecast_metric(
            actual,
            predicted,
            y_train=train,
            metric_name='mase',
            seasonal_period=seasonal_period,
        )
        baseline_smape = _evaluate_forecast_metric(
            actual,
            baseline,
            y_train=train,
            metric_name='smape',
            seasonal_period=seasonal_period,
        )
        baseline_mase = _evaluate_forecast_metric(
            actual,
            baseline,
            y_train=train,
            metric_name='mase',
            seasonal_period=seasonal_period,
        )
        baseline_smape_value = baseline_smape.metric_value if baseline_smape.metric_value > 1e-8 else 1.0
        baseline_mase_value = baseline_mase.metric_value if baseline_mase.metric_value > 1e-8 else 1.0
        smape_ratio = np.asarray(smape_eval.per_horizon_metrics, dtype=float) / baseline_smape_value
        mase_ratio = np.asarray(mase_eval.per_horizon_metrics, dtype=float) / baseline_mase_value
        per_horizon = 0.5 * (smape_ratio + mase_ratio)
        metric_value = float(np.mean(per_horizon))
    else:
        raise ValueError(f'Unsupported runtime forecasting metric: {metric_name}')

    return ForecastingEvaluationResult(
        metric_name=metric,
        metric_value=metric_value,
        per_horizon_metrics=tuple(float(value) for value in np.asarray(per_horizon, dtype=float).tolist()),
        metadata={'n_horizons': int(len(actual)), 'seasonal_period': int(seasonal_period)},
    )


def _instantiate_runtime_model(
        canonical_model_name: str,
        *,
        forecast_horizon: int,
        params: dict[str, Any],
        series_length: int | None = None,
):
    if canonical_model_name == 'lagged_ridge_forecaster':
        from .lagged_ridge_forecaster import LaggedRidgeForecaster
        model_cls = LaggedRidgeForecaster
    elif canonical_model_name == 'low_rank_lagged_ridge_forecaster':
        from .low_rank_lagged_ridge_forecaster import LowRankLaggedRidgeForecaster
        model_cls = LowRankLaggedRidgeForecaster
    elif canonical_model_name in {'okhs', 'okhs_fdmd_forecaster'}:
        if build_okhs_fdmd_forecaster is None:
            raise ValueError('OKHS FDMD runtime is unavailable in the current environment.')
        return build_okhs_fdmd_forecaster(
            forecast_horizon=int(forecast_horizon),
            params=dict(params),
            series_length=series_length,
        )
    elif canonical_model_name == 'havok_forecaster':
        from .havok_forecaster import HAVOKForecaster
        model_cls = HAVOKForecaster
    elif canonical_model_name == 'mssa_forecaster':
        from .mssa_forecaster import MSSAForecaster
        model_cls = MSSAForecaster
    elif canonical_model_name == 'ssa_forecaster':
        from .mssa_forecaster import MSSAForecaster
        params = {**params, 'coupled': False}
        model_cls = MSSAForecaster
    elif canonical_model_name == 'hybrid_ensemble_forecaster':
        from .hybrid_ensemble_forecaster import HybridEnsembleForecaster
        model_cls = HybridEnsembleForecaster
    elif canonical_model_name in {'patch_tst_model', 'tcn_model', 'deepar_model', 'nbeats_model'}:
        if build_neural_forecast_head is None:
            raise ValueError('Neural forecast head runtime is unavailable in the current environment.')
        return build_neural_forecast_head(
            canonical_model_name,
            forecast_horizon=int(forecast_horizon),
            params=dict(params),
        )
    elif canonical_model_name == 'lagged_forecaster':
        channel_model = str(params.get('channel_model', 'ridge')).lower()
        if channel_model != 'ridge':
            raise ValueError(
                "Runtime stage tuning supports 'lagged_forecaster' only with channel_model='ridge'."
            )
        from .lagged_ridge_forecaster import LaggedRidgeForecaster
        model_cls = LaggedRidgeForecaster
    else:
        raise ValueError(f'Unsupported runtime stage tuning model: {canonical_model_name}')

    signature = inspect.signature(model_cls)
    accepted_params = {
        key: value
        for key, value in dict(params).items()
        if key in signature.parameters and key != 'forecast_horizon'
    }
    return model_cls(forecast_horizon=int(forecast_horizon), **accepted_params)


def evaluate_forecasting_model_on_series(
        model_name: str,
        *,
        time_series: np.ndarray,
        forecast_horizon: int,
        params: dict[str, Any] | None = None,
        metric_name: str = 'rmse',
        split_spec: ForecastingSplitSpec | None = None,
        seasonal_period: int = 1,
) -> ForecastingSeriesEvaluation:
    canonical_model_name = canonical_forecasting_model_name(model_name)
    resolved_params = _normalize_base_params(params, model_name=canonical_model_name)
    device_policy = TensorDevicePolicy(device=str(resolved_params.get('device', 'cpu')))
    full_series = _prepare_runtime_series(canonical_model_name, np.asarray(time_series, dtype=float), resolved_params)
    batch = series_to_forecast_tensor_batch(
        full_series,
        forecast_horizon=int(forecast_horizon),
        device_policy=device_policy,
        metadata={'model_name': canonical_model_name},
    )
    resolved_split = split_spec or ForecastingSplitSpec(validation_horizon=int(forecast_horizon))
    train_batch, validation_target = split_forecasting_batch(batch, resolved_split)
    validation_horizon = int(resolved_split.validation_horizon or forecast_horizon)
    train_series = train_batch.history.detach().cpu().numpy()
    prepared_train_series = _prepare_runtime_series(canonical_model_name, train_series, resolved_params)

    model = _instantiate_runtime_model(
        canonical_model_name,
        forecast_horizon=validation_horizon,
        params=resolved_params,
        series_length=int(np.asarray(prepared_train_series).reshape(-1).shape[0]),
    )
    model.fit(prepared_train_series)
    try:
        raw_forecast = model.predict(prepared_train_series, forecast_horizon=validation_horizon)
    except TypeError:
        raw_forecast = model.predict(prepared_train_series)
    forecast = np.asarray(raw_forecast, dtype=float).reshape(-1)[:validation_horizon]
    target = validation_target.detach().cpu().numpy().reshape(-1)[:validation_horizon]
    evaluation = _evaluate_forecast_metric(
        target,
        forecast,
        y_train=np.asarray(prepared_train_series, dtype=float),
        metric_name=metric_name,
        seasonal_period=int(seasonal_period),
    )
    diagnostics = model.get_diagnostics() if hasattr(model, 'get_diagnostics') else {}
    family = str(diagnostics.get('model_family', 'forecasting'))
    return ForecastingSeriesEvaluation(
        model_name=model_name,
        canonical_model_name=canonical_model_name,
        family=family,
        parameters=resolved_params,
        forecast_horizon=validation_horizon,
        metric=evaluation,
        forecast=tuple(float(value) for value in forecast.tolist()),
        target=tuple(float(value) for value in target.tolist()),
        split_metadata={
            'split_kind': resolved_split.kind.value,
            'train_length': int(train_batch.history.shape[0]),
            'validation_horizon': int(validation_horizon),
            'seasonal_period': int(seasonal_period),
        },
        diagnostics=dict(diagnostics),
        metadata={'device': str(device_policy.resolve_device())},
    )


def build_forecasting_stage_objective_from_series(
        model_name: str,
        *,
        time_series: np.ndarray,
        forecast_horizon: int,
        metric_name: str = 'rmse',
        split_spec: ForecastingSplitSpec | None = None,
        seasonal_period: int = 1,
) -> Callable[[dict[str, Any]], float]:
    def objective(candidate_params: dict[str, Any]) -> float:
        evaluation = evaluate_forecasting_model_on_series(
            model_name,
            time_series=time_series,
            forecast_horizon=forecast_horizon,
            params=candidate_params,
            metric_name=metric_name,
            split_spec=split_spec,
            seasonal_period=seasonal_period,
        )
        return float(evaluation.metric.metric_value)

    return objective


def run_forecasting_stage_tuning_on_series(
        model_name: str,
        *,
        time_series: np.ndarray,
        forecast_horizon: int,
        base_params: dict[str, Any] | None = None,
        stage_updates: dict[str, Any] | None = None,
        metric_name: str = 'rmse',
        split_spec: ForecastingSplitSpec | None = None,
        seasonal_period: int = 1,
        max_values_per_parameter: int = 3,
        max_stage_candidates: int = 16,
) -> ForecastingSeriesStageTuningResult:
    resolved_params = _normalize_base_params(base_params, model_name=model_name)
    objective = build_forecasting_stage_objective_from_series(
        model_name,
        time_series=time_series,
        forecast_horizon=forecast_horizon,
        metric_name=metric_name,
        split_spec=split_spec,
        seasonal_period=seasonal_period,
    )
    baseline_evaluation = evaluate_forecasting_model_on_series(
        model_name,
        time_series=time_series,
        forecast_horizon=forecast_horizon,
        params=resolved_params,
        metric_name=metric_name,
        split_spec=split_spec,
        seasonal_period=seasonal_period,
    )
    sequential_result = run_sequential_stage_tuning(
        model_name,
        objective=objective,
        base_params=resolved_params,
        stage_updates=stage_updates,
        max_values_per_parameter=max_values_per_parameter,
        max_stage_candidates=max_stage_candidates,
    )
    best_evaluation = evaluate_forecasting_model_on_series(
        model_name,
        time_series=time_series,
        forecast_horizon=forecast_horizon,
        params=sequential_result.best_parameters,
        metric_name=metric_name,
        split_spec=split_spec,
        seasonal_period=seasonal_period,
    )
    return ForecastingSeriesStageTuningResult(
        model_name=model_name,
        canonical_model_name=sequential_result.canonical_model_name,
        family=sequential_result.family,
        sequential_result=sequential_result,
        baseline_evaluation=baseline_evaluation,
        best_evaluation=best_evaluation,
        metadata={
            'metric_name': metric_name,
            'baseline_score': float(baseline_evaluation.metric.metric_value),
            'best_score': float(best_evaluation.metric.metric_value),
            'improved': bool(best_evaluation.metric.metric_value <= baseline_evaluation.metric.metric_value),
        },
    )
