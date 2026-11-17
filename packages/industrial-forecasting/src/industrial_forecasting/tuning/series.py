"""Evaluate and tune a forecasting model using fold-local training and explicit factories."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from ..metrics import ForecastMetric, score_forecast

from ..neural.config import (
    normalize_neural_forecasting_params,
)
from .execution import (
    ForecastingSequentialStageTuningResult,
    run_sequential_stage_tuning,
)
from ..neural.runtime import (
    ForecastingSplitKind,
    ForecastingEvaluationResult,
    ForecastingSplitSpec,
    TensorDevicePolicy,
    iter_forecasting_splits,
    series_to_forecast_tensor_batch,
)
from ..progress import (
    ForecastingProgressPolicy,
    resolve_forecasting_progress_policy,
)
from ..model_names import canonical_forecasting_model_name
from ..neural.models import SeriesModelFactory, build_series_model


@dataclass(frozen=True)
class ForecastingSeriesEvaluation:
    """Evaluation result for one forecasting model on one time series."""

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
        """Serialize the series-level evaluation payload."""
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
    """Baseline-vs-tuned evaluation result for one time series."""

    model_name: str
    canonical_model_name: str
    family: str
    sequential_result: ForecastingSequentialStageTuningResult
    baseline_evaluation: ForecastingSeriesEvaluation
    best_evaluation: ForecastingSeriesEvaluation
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize baseline, tuning and best-evaluation details."""
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
    elif canonical_name == 'topo_forecaster':
        normalized.setdefault('channel_model', 'ridge')
        normalized.setdefault('window_size', 10)
        normalized.setdefault('patch_len', 10)
        normalized.setdefault('stride', 1)
        normalized.setdefault('alpha', 1.0)
        normalized.setdefault('device', 'auto')
    elif canonical_name in {'mssa_forecaster', 'ssa_forecaster'}:
        normalized.setdefault('head_policy', 'mlp')
        normalized.setdefault('head_hidden_dim', 64)
        normalized.setdefault('head_hidden_layers', 2)
        normalized.setdefault('head_epochs', 120)
        normalized.setdefault('head_learning_rate', 1e-3)
        normalized.setdefault('device', 'auto')
    elif canonical_name == 'havok_forecaster':
        if 'head_activation' not in normalized and normalized.get('activation') is not None:
            normalized['head_activation'] = normalized['activation']
        if 'head_depth' not in normalized and normalized.get('head_hidden_layers') is not None:
            normalized['head_depth'] = normalized['head_hidden_layers']
        if 'head_base_hidden_dim' not in normalized and normalized.get('head_hidden_dim') is not None:
            normalized['head_base_hidden_dim'] = normalized['head_hidden_dim']
        normalized.setdefault('head_policy', 'mlp')
        normalized.setdefault('head_activation', 'relu')
        normalized.setdefault('head_depth', 2)
        normalized.setdefault('head_base_hidden_dim', 512)
        normalized.setdefault('head_epochs', 120)
        normalized.setdefault('head_learning_rate', 1e-3)
        normalized.setdefault('device', 'auto')
    elif canonical_name in {'patch_tst_model', 'tst_model', 'tcn_model', 'deepar_model', 'nbeats_model'}:
        normalized = normalize_neural_forecasting_params(normalized)
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


@dataclass(frozen=True)
class ForecastMetricEvaluator:
    """Evaluate one temporal fold with access to its training history."""

    y_true: np.ndarray
    y_pred: np.ndarray
    y_train: np.ndarray
    seasonal_period: int

    def evaluate(self, metric_name: str) -> ForecastingEvaluationResult:
        metric = str(metric_name).lower()
        try:
            name = ForecastMetric(metric)
        except ValueError as error:
            raise ValueError(f'Unsupported runtime forecasting metric: {metric_name}') from error
        score = score_forecast(
            name,
            np.asarray(self.y_true, dtype=float).reshape(-1),
            np.asarray(self.y_pred, dtype=float).reshape(-1),
            history=np.asarray(self.y_train, dtype=float).reshape(-1),
            seasonal_period=max(1, self.seasonal_period),
        )
        metadata = {'n_horizons': len(score.by_step)}
        if name not in (ForecastMetric.MAE, ForecastMetric.RMSE):
            metadata['seasonal_period'] = int(self.seasonal_period)
        return ForecastingEvaluationResult(
            metric_name=metric, metric_value=score.value,
            per_horizon_metrics=tuple(score.by_step.tolist()), metadata=metadata,
        )


def _evaluate_forecast_metric(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        y_train: np.ndarray,
        metric_name: str,
        seasonal_period: int,
) -> ForecastingEvaluationResult:
    return ForecastMetricEvaluator(
        y_true=y_true,
        y_pred=y_pred,
        y_train=y_train,
        seasonal_period=int(seasonal_period),
    ).evaluate(metric_name)


def _aggregate_fold_metrics(evaluations: list[ForecastingEvaluationResult]) -> ForecastingEvaluationResult:
    if not evaluations:
        raise ValueError('At least one fold evaluation is required.')
    metric_name = str(evaluations[0].metric_name)
    metric_values = np.asarray([evaluation.metric_value for evaluation in evaluations], dtype=float)
    per_fold_vectors = [np.asarray(evaluation.per_horizon_metrics, dtype=float) for evaluation in evaluations]
    max_length = max(len(vector) for vector in per_fold_vectors)
    padded = np.full((len(per_fold_vectors), max_length), np.nan, dtype=float)
    for index, vector in enumerate(per_fold_vectors):
        padded[index, :len(vector)] = vector
    aggregated_per_horizon = np.nanmean(padded, axis=0)
    return ForecastingEvaluationResult(
        metric_name=metric_name,
        metric_value=float(np.mean(metric_values)),
        per_horizon_metrics=tuple(float(value) for value in aggregated_per_horizon.tolist()),
        metadata={
            'fold_count': int(len(evaluations)),
            'aggregation': 'mean_across_folds',
            'per_fold_metric_values': [float(value) for value in metric_values.tolist()],
        },
    )


@dataclass
class ForecastingSeriesEvaluator:
    """Fit, forecast and score a model across temporal CV folds."""

    model_name: str
    time_series: np.ndarray
    forecast_horizon: int
    params: dict[str, Any] | None = None
    metric_name: str = 'rmse'
    split_spec: ForecastingSplitSpec | None = None
    seasonal_period: int = 1
    show_progress: bool | None = None
    progress_policy: ForecastingProgressPolicy | dict[str, Any] | bool | None = None
    model_factory: SeriesModelFactory = field(default=build_series_model, kw_only=True)

    def __post_init__(self):
        """Resolve model name, parameters, progress policy and split spec."""
        self.canonical_model_name = canonical_forecasting_model_name(self.model_name)
        self.resolved_params = _normalize_base_params(self.params, model_name=self.canonical_model_name)
        self.resolved_progress_policy = resolve_forecasting_progress_policy(
            self.progress_policy,
            show_progress=self.show_progress,
        )
        if self.canonical_model_name in {'mssa_forecaster', 'ssa_forecaster', 'havok_forecaster'}:
            self.resolved_params['progress_policy'] = self.resolved_progress_policy
        self.device_policy = TensorDevicePolicy(device=str(self.resolved_params.get('device', 'auto')))
        self.full_series = _prepare_runtime_series(
            self.canonical_model_name,
            np.asarray(self.time_series, dtype=float),
            self.resolved_params,
        )
        self.batch = series_to_forecast_tensor_batch(
            self.full_series,
            forecast_horizon=int(self.forecast_horizon),
            device_policy=self.device_policy,
            metadata={'model_name': self.canonical_model_name},
        )
        self.resolved_split = self.split_spec or ForecastingSplitSpec(
            kind=ForecastingSplitKind.TIME_SERIES_SPLIT,
            validation_horizon=int(self.forecast_horizon),
            n_splits=3,
            gap=0,
        )
        self.validation_horizon = int(
            self.resolved_split.test_size or self.resolved_split.validation_horizon or self.forecast_horizon
        )

    def _iter_over_folds(self):
        return iter_forecasting_splits(self.batch, self.resolved_split)

    def _build_runtime_model(self, prepared_train_series: np.ndarray):
        return self.model_factory(
            self.canonical_model_name,
            forecast_horizon=self.validation_horizon,
            params=self.resolved_params,
            series_length=int(np.asarray(prepared_train_series).reshape(-1).shape[0]),
        )

    def _fit_predict_fold(
            self,
            fold,
    ) -> tuple[np.ndarray, np.ndarray, ForecastingEvaluationResult, dict[str, Any], str]:
        train_series = fold.train_batch.history.detach().cpu().numpy()
        prepared_train_series = _prepare_runtime_series(
            self.canonical_model_name,
            train_series,
            self.resolved_params,
        )
        model = self._build_runtime_model(prepared_train_series)
        model.fit(prepared_train_series)
        try:
            raw_forecast = model.predict(prepared_train_series, forecast_horizon=self.validation_horizon)
        except TypeError:
            raw_forecast = model.predict(prepared_train_series)
        forecast = np.asarray(raw_forecast, dtype=float).reshape(-1)[:self.validation_horizon]
        target = fold.validation_target.detach().cpu().numpy().reshape(-1)[:self.validation_horizon]
        fold_metric = _evaluate_forecast_metric(
            target,
            forecast,
            y_train=np.asarray(prepared_train_series, dtype=float),
            metric_name=self.metric_name,
            seasonal_period=int(self.seasonal_period),
        )
        diagnostics = model.get_diagnostics() if hasattr(model, 'get_diagnostics') else {}
        family = str(diagnostics.get('model_family', 'forecasting'))
        return forecast, target, fold_metric, diagnostics, family

    def _build_fold_detail(self, fold, fold_metric: ForecastingEvaluationResult) -> dict[str, Any]:
        return {
            **fold.to_dict(),
            'metric': fold_metric.to_dict(),
        }

    def _build_split_metadata(self, folds, fold_details: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            'split_kind': self.resolved_split.kind.value,
            'train_length': int(folds[-1].train_batch.history.shape[0]),
            'train_lengths': [int(fold.train_batch.history.shape[0]) for fold in folds],
            'fold_count': int(len(folds)),
            'validation_horizon': int(self.validation_horizon),
            'seasonal_period': int(self.seasonal_period),
            'gap': int(self.resolved_split.gap or 0),
            'n_splits': int(self.resolved_split.n_splits or len(folds)),
            'folds': fold_details,
        }

    def run(self) -> ForecastingSeriesEvaluation:
        """Execute all folds and aggregate the resulting forecast metrics."""
        folds = self._iter_over_folds()
        fold_evaluations: list[ForecastingEvaluationResult] = []
        fold_forecasts: list[np.ndarray] = []
        fold_targets: list[np.ndarray] = []
        fold_details: list[dict[str, Any]] = []
        diagnostics: dict[str, Any] = {}
        family = 'forecasting'

        for fold in folds:
            forecast, target, fold_metric, diagnostics, family = self._fit_predict_fold(fold)
            fold_evaluations.append(fold_metric)
            fold_forecasts.append(forecast)
            fold_targets.append(target)
            fold_details.append(self._build_fold_detail(fold, fold_metric))

        evaluation = _aggregate_fold_metrics(fold_evaluations)
        forecast = np.concatenate(fold_forecasts) if fold_forecasts else np.asarray([], dtype=float)
        target = np.concatenate(fold_targets) if fold_targets else np.asarray([], dtype=float)
        return ForecastingSeriesEvaluation(
            model_name=self.model_name,
            canonical_model_name=self.canonical_model_name,
            family=family,
            parameters=self.resolved_params,
            forecast_horizon=self.validation_horizon,
            metric=evaluation,
            forecast=tuple(float(value) for value in forecast.tolist()),
            target=tuple(float(value) for value in target.tolist()),
            split_metadata=self._build_split_metadata(folds, fold_details),
            diagnostics=dict(diagnostics),
            metadata={
                'device': str(self.device_policy.resolve_device()),
                'progress_policy': self.resolved_progress_policy.to_dict(),
            },
        )


@dataclass
class ForecastingSeriesStageTuningRunner:
    """Run baseline evaluation, sequential tuning and best-model evaluation."""

    model_name: str
    time_series: np.ndarray
    forecast_horizon: int
    base_params: dict[str, Any] | None = None
    stage_updates: dict[str, Any] | None = None
    metric_name: str = 'rmse'
    split_spec: ForecastingSplitSpec | None = None
    seasonal_period: int = 1
    max_values_per_parameter: int = 3
    max_stage_candidates: int = 16
    show_progress: bool | None = None
    progress_policy: ForecastingProgressPolicy | dict[str, Any] | bool | None = None
    model_factory: SeriesModelFactory = field(default=build_series_model, kw_only=True)
    parameter_space: Mapping[str, Any] | None = field(default=None, kw_only=True)

    def __post_init__(self):
        """Resolve base parameters and progress policy before tuning."""
        self.resolved_progress_policy = resolve_forecasting_progress_policy(
            self.progress_policy,
            show_progress=self.show_progress,
        )
        self.resolved_params = _normalize_base_params(self.base_params, model_name=self.model_name)
        if canonical_forecasting_model_name(self.model_name) in {'mssa_forecaster', 'ssa_forecaster',
                                                                 'havok_forecaster'}:
            self.resolved_params['progress_policy'] = self.resolved_progress_policy

    def _evaluate(self, params: dict[str, Any]) -> ForecastingSeriesEvaluation:
        return ForecastingSeriesEvaluator(
            self.model_name,
            time_series=self.time_series,
            forecast_horizon=self.forecast_horizon,
            params=params,
            metric_name=self.metric_name,
            split_spec=self.split_spec,
            seasonal_period=self.seasonal_period,
            show_progress=self.show_progress,
            progress_policy=self.resolved_progress_policy,
            model_factory=self.model_factory,
        ).run()

    def _build_objective(self) -> Callable[[dict[str, Any]], float]:
        def objective(candidate_params: dict[str, Any]) -> float:
            evaluation = self._evaluate(candidate_params)
            return float(evaluation.metric.metric_value)

        return objective

    def _run_sequential(self) -> ForecastingSequentialStageTuningResult:
        return run_sequential_stage_tuning(
            self.model_name,
            objective=self._build_objective(),
            base_params=self.resolved_params,
            stage_updates=self.stage_updates,
            max_values_per_parameter=self.max_values_per_parameter,
            max_stage_candidates=self.max_stage_candidates,
            show_progress=self.show_progress,
            progress_policy=self.resolved_progress_policy,
            parameter_space=self.parameter_space,
        )

    def run(self) -> ForecastingSeriesStageTuningResult:
        """Compare baseline parameters with stage-tuned parameters on a series."""
        baseline_evaluation = self._evaluate(self.resolved_params)
        sequential_result = self._run_sequential()
        best_evaluation = self._evaluate(sequential_result.best_parameters)
        return ForecastingSeriesStageTuningResult(
            model_name=self.model_name,
            canonical_model_name=sequential_result.canonical_model_name,
            family=sequential_result.family,
            sequential_result=sequential_result,
            baseline_evaluation=baseline_evaluation,
            best_evaluation=best_evaluation,
            metadata={
                'metric_name': self.metric_name,
                'baseline_score': float(baseline_evaluation.metric.metric_value),
                'best_score': float(best_evaluation.metric.metric_value),
                'improved': bool(best_evaluation.metric.metric_value <= baseline_evaluation.metric.metric_value),
                'progress_policy': self.resolved_progress_policy.to_dict(),
            },
        )
