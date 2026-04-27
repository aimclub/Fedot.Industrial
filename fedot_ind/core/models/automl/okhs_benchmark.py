from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from statistics import NormalDist
from typing import Any, Callable

import numpy as np

from fedot_ind.core.models.kernel.okhs_common import OKHSMethod, QPolicy
from fedot_ind.core.models.kernel.okhs_forecasting import OKHSForecaster
from fedot_ind.core.operation.transformation.representation.kernel.kernels import DataDrivenQSelector

ModelFactory = Callable[[], Any]
ModelFitFn = Callable[[Any, np.ndarray], None]
ModelPredictFn = Callable[[Any, np.ndarray, int], np.ndarray]
ModelMetadataFn = Callable[[Any], dict[str, Any]]


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    series: np.ndarray
    forecast_horizon: int
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class BenchmarkModelSpec:
    name: str
    factory: ModelFactory
    fit_fn: ModelFitFn
    predict_fn: ModelPredictFn
    tags: tuple[str, ...] = ()
    metadata_fn: ModelMetadataFn | None = None


@dataclass(frozen=True)
class BenchmarkRunResult:
    dataset_name: str
    model_name: str
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    forecast_horizon: int
    tags: tuple[str, ...] = ()
    validation_metadata: dict[str, Any] | None = None
    test_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    datasets: tuple[BenchmarkDataset, ...]
    model_specs: tuple[BenchmarkModelSpec, ...]
    metrics: tuple[str, ...] = ('mae', 'rmse')
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    seed: int = 0


@dataclass(frozen=True)
class BenchmarkReport:
    config: BenchmarkConfig
    runs: tuple[BenchmarkRunResult, ...]
    leaderboard: tuple[dict[str, float | str], ...]


@dataclass(frozen=True)
class QCandidateScore:
    q: float
    metric_value: float


@dataclass(frozen=True)
class QDecisionTrace:
    policy: str
    selected_q: float
    reason: str
    candidate_scores: tuple[QCandidateScore, ...] = ()
    diagnostics: dict[str, float | str] | None = None


@dataclass(frozen=True)
class QSelectionSpec:
    policy: str | QPolicy
    fixed_q: float = 0.7
    q_grid: tuple[float, ...] = ()
    selection_metric: str = 'mae'
    validation_fraction: float = 0.25
    selector: Any | None = None


class ContextWindowPolicy(str, Enum):
    EXPANDING = 'expanding'
    ROLLING = 'rolling'


class RefitPolicy(str, Enum):
    NEVER = 'never'
    ALWAYS = 'always'
    PERIODIC = 'periodic'
    DRIFT = 'drift'


@dataclass(frozen=True)
class RollingForecastConfig:
    forecast_horizon: int
    step_size: int = 1
    context_policy: ContextWindowPolicy = ContextWindowPolicy.EXPANDING
    rolling_window_size: int | None = None
    refit_policy: RefitPolicy = RefitPolicy.ALWAYS
    update_frequency: int = 1
    drift_threshold: float | None = None
    metrics: tuple[str, ...] = ('mae',)


@dataclass(frozen=True)
class RollingForecastStep:
    step_index: int
    train_end_index: int
    horizon: int
    metrics: dict[str, float]
    actual_values: tuple[float, ...]
    forecast_values: tuple[float, ...]
    refit_performed: bool
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RollingForecastReport:
    model_name: str
    config: RollingForecastConfig
    steps: tuple[RollingForecastStep, ...]
    aggregate_metrics: dict[str, float]


class UncertaintyMethod(str, Enum):
    RESIDUAL_STD = 'residual_std'


@dataclass(frozen=True)
class UncertaintyConfig:
    confidence_level: float = 0.95
    method: UncertaintyMethod = UncertaintyMethod.RESIDUAL_STD
    min_history: int = 3
    width_warning_threshold: float = 5.0
    error_floor: float = 1e-8


@dataclass(frozen=True)
class ForecastInterval:
    lower: tuple[float, ...]
    center: tuple[float, ...]
    upper: tuple[float, ...]


@dataclass(frozen=True)
class UncertaintyStep:
    step_index: int
    interval: ForecastInterval
    residual_scale: float
    empirical_coverage_proxy: float
    quality_flags: tuple[str, ...]


@dataclass(frozen=True)
class UncertaintyReport:
    model_name: str
    config: UncertaintyConfig
    steps: tuple[UncertaintyStep, ...]
    diagnostics: dict[str, float | str]


def validate_benchmark_config(config: BenchmarkConfig) -> None:
    if not config.datasets:
        raise ValueError('BenchmarkConfig must include at least one dataset.')
    if not config.model_specs:
        raise ValueError('BenchmarkConfig must include at least one model spec.')
    if config.validation_fraction <= 0 or config.test_fraction <= 0:
        raise ValueError('validation_fraction and test_fraction must be positive.')
    if config.validation_fraction + config.test_fraction >= 1:
        raise ValueError('validation_fraction + test_fraction must be less than 1.')

    supported_metrics = {'mae', 'rmse'}
    unsupported = set(config.metrics) - supported_metrics
    if unsupported:
        raise ValueError(f'Unsupported benchmark metrics: {sorted(unsupported)}')

    for dataset in config.datasets:
        if dataset.forecast_horizon <= 0:
            raise ValueError(f'Dataset {dataset.name} must have a positive forecast_horizon.')
        if len(dataset.series) <= dataset.forecast_horizon:
            raise ValueError(
                f'Dataset {dataset.name} must be longer than its forecast_horizon.'
            )


def build_holdout_split(
        series: np.ndarray,
        validation_fraction: float,
        test_fraction: float,
) -> BenchmarkSplit:
    n_samples = len(series)
    validation_size = max(1, int(round(n_samples * validation_fraction)))
    test_size = max(1, int(round(n_samples * test_fraction)))
    train_size = n_samples - validation_size - test_size

    if train_size < 2:
        raise ValueError('Holdout split leaves too few training samples.')

    train = np.asarray(series[:train_size], dtype=float)
    validation = np.asarray(series[train_size:train_size + validation_size], dtype=float)
    test = np.asarray(series[train_size + validation_size:], dtype=float)
    return BenchmarkSplit(train=train, validation=validation, test=test)


def compute_metric(metric_name: str, actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) != len(predicted):
        raise ValueError('Actual and predicted arrays must have the same length.')

    if metric_name == 'mae':
        return float(np.mean(np.abs(actual - predicted)))
    if metric_name == 'rmse':
        return float(np.sqrt(np.mean((actual - predicted) ** 2)))
    raise ValueError(f'Unsupported metric: {metric_name}')


def _truncate_forecast(predictions: np.ndarray, horizon: int) -> np.ndarray:
    forecast = np.asarray(predictions, dtype=float).reshape(-1)
    if len(forecast) < horizon:
        raise ValueError(
            f'Model returned only {len(forecast)} predictions for horizon {horizon}.'
        )
    return forecast[:horizon]


def _score_forecast(
        model_spec: BenchmarkModelSpec,
        context_series: np.ndarray,
        target_series: np.ndarray,
        metrics: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, Any]]:
    model = model_spec.factory()
    model_spec.fit_fn(model, context_series)
    forecast = _truncate_forecast(
        model_spec.predict_fn(model, context_series, len(target_series)),
        len(target_series),
    )
    metric_values = {
        metric_name: compute_metric(metric_name, target_series, forecast)
        for metric_name in metrics
    }
    metadata = model_spec.metadata_fn(model) if model_spec.metadata_fn else {}
    return metric_values, metadata


def build_leaderboard(
        runs: tuple[BenchmarkRunResult, ...],
        primary_metric: str = 'mae',
) -> tuple[dict[str, float | str], ...]:
    ordered = sorted(
        runs,
        key=lambda run: run.test_metrics[primary_metric],
    )
    return tuple(
        {
            'dataset_name': run.dataset_name,
            'model_name': run.model_name,
            primary_metric: run.test_metrics[primary_metric],
        }
        for run in ordered
    )


def run_benchmark(config: BenchmarkConfig) -> BenchmarkReport:
    validate_benchmark_config(config)

    runs: list[BenchmarkRunResult] = []
    for dataset in config.datasets:
        split = build_holdout_split(
            np.asarray(dataset.series, dtype=float),
            validation_fraction=config.validation_fraction,
            test_fraction=config.test_fraction,
        )

        for model_spec in config.model_specs:
            validation_metrics, validation_metadata = _score_forecast(
                model_spec=model_spec,
                context_series=split.train,
                target_series=split.validation,
                metrics=config.metrics,
            )
            test_context = np.concatenate([split.train, split.validation])
            test_metrics, test_metadata = _score_forecast(
                model_spec=model_spec,
                context_series=test_context,
                target_series=split.test,
                metrics=config.metrics,
            )
            runs.append(
                BenchmarkRunResult(
                    dataset_name=dataset.name,
                    model_name=model_spec.name,
                    validation_metrics=validation_metrics,
                    test_metrics=test_metrics,
                    forecast_horizon=dataset.forecast_horizon,
                    tags=dataset.tags + model_spec.tags,
                    validation_metadata=validation_metadata,
                    test_metadata=test_metadata,
                )
            )

    runs_tuple = tuple(runs)
    return BenchmarkReport(
        config=config,
        runs=runs_tuple,
        leaderboard=build_leaderboard(runs_tuple, primary_metric=config.metrics[0]),
    )


def benchmark_report_to_dict(report: BenchmarkReport) -> dict[str, Any]:
    return {
        'config': {
            'datasets': [
                {
                    'name': dataset.name,
                    'forecast_horizon': dataset.forecast_horizon,
                    'tags': list(dataset.tags),
                    'series_length': int(len(dataset.series)),
                }
                for dataset in report.config.datasets
            ],
            'model_specs': [
                {
                    'name': model_spec.name,
                    'tags': list(model_spec.tags),
                }
                for model_spec in report.config.model_specs
            ],
            'metrics': list(report.config.metrics),
            'validation_fraction': report.config.validation_fraction,
            'test_fraction': report.config.test_fraction,
            'seed': report.config.seed,
        },
        'runs': [asdict(run) for run in report.runs],
        'leaderboard': [dict(row) for row in report.leaderboard],
    }


def _make_json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_make_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_make_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _make_json_ready(item) for key, item in value.items()}
    return value


def render_benchmark_markdown(report: BenchmarkReport) -> str:
    lines = [
        '# OKHS Benchmark Report',
        '',
        f"- datasets: {len(report.config.datasets)}",
        f"- models: {len(report.config.model_specs)}",
        f"- metrics: {', '.join(report.config.metrics)}",
        '',
        '## Leaderboard',
    ]

    for row in report.leaderboard:
        metric_name = report.config.metrics[0]
        lines.append(
            f"- {row['dataset_name']} / {row['model_name']}: {metric_name}={row[metric_name]:.6f}"
        )

    lines.append('')
    lines.append('## Runs')
    for run in report.runs:
        validation_summary = ', '.join(
            f"{name}={value:.6f}" for name, value in run.validation_metrics.items()
        )
        test_summary = ', '.join(
            f"{name}={value:.6f}" for name, value in run.test_metrics.items()
        )
        lines.append(
            f"- {run.dataset_name} / {run.model_name}: "
            f"validation[{validation_summary}] test[{test_summary}]"
        )
        if run.test_metadata:
            selected_q = run.test_metadata.get('selected_q')
            policy = run.test_metadata.get('policy')
            if selected_q is not None and policy is not None:
                lines.append(
                    f"  q-selection: policy={policy}, selected_q={float(selected_q):.6f}"
                )

    return '\n'.join(lines)


def validate_rolling_forecast_config(config: RollingForecastConfig) -> None:
    if config.forecast_horizon <= 0:
        raise ValueError('forecast_horizon must be positive.')
    if config.step_size <= 0:
        raise ValueError('step_size must be positive.')
    if config.update_frequency <= 0:
        raise ValueError('update_frequency must be positive.')
    if config.context_policy is ContextWindowPolicy.ROLLING:
        if config.rolling_window_size is None or config.rolling_window_size < 2:
            raise ValueError('rolling_window_size must be set to at least 2 for rolling context.')
    if config.refit_policy is RefitPolicy.DRIFT and config.drift_threshold is None:
        raise ValueError('drift_threshold is required when refit_policy="drift".')
    supported_metrics = {'mae', 'rmse'}
    unsupported = set(config.metrics) - supported_metrics
    if unsupported:
        raise ValueError(f'Unsupported rolling metrics: {sorted(unsupported)}')


def validate_uncertainty_config(config: UncertaintyConfig) -> None:
    if not (0 < config.confidence_level < 1):
        raise ValueError('confidence_level must be between 0 and 1.')
    if config.min_history < 1:
        raise ValueError('min_history must be positive.')
    if config.width_warning_threshold <= 0:
        raise ValueError('width_warning_threshold must be positive.')
    if config.error_floor < 0:
        raise ValueError('error_floor must be non-negative.')


def _build_context_slice(
        series: np.ndarray,
        train_end_index: int,
        config: RollingForecastConfig,
) -> np.ndarray:
    if config.context_policy is ContextWindowPolicy.EXPANDING:
        return np.asarray(series[:train_end_index], dtype=float)

    assert config.rolling_window_size is not None
    start_index = max(0, train_end_index - config.rolling_window_size)
    return np.asarray(series[start_index:train_end_index], dtype=float)


def _should_refit_model(
        step_index: int,
        previous_error: float | None,
        config: RollingForecastConfig,
) -> bool:
    if step_index == 0:
        return True
    if config.refit_policy is RefitPolicy.ALWAYS:
        return True
    if config.refit_policy is RefitPolicy.NEVER:
        return False
    if config.refit_policy is RefitPolicy.PERIODIC:
        return step_index % config.update_frequency == 0
    assert config.drift_threshold is not None
    return previous_error is not None and previous_error > config.drift_threshold


def _aggregate_step_metrics(steps: tuple[RollingForecastStep, ...]) -> dict[str, float]:
    metric_names = steps[0].metrics.keys()
    return {
        metric_name: float(np.mean([step.metrics[metric_name] for step in steps]))
        for metric_name in metric_names
    }


def run_rolling_forecast(
        series: np.ndarray,
        model_spec: BenchmarkModelSpec,
        config: RollingForecastConfig,
) -> RollingForecastReport:
    validate_rolling_forecast_config(config)

    normalized_series = np.asarray(series, dtype=float).reshape(-1)
    if len(normalized_series) <= config.forecast_horizon:
        raise ValueError('Series must be longer than forecast_horizon for rolling evaluation.')

    minimum_context = (
        config.rolling_window_size
        if config.context_policy is ContextWindowPolicy.ROLLING
        else max(config.forecast_horizon, 2)
    )
    if minimum_context is None:
        minimum_context = max(config.forecast_horizon, 2)
    if len(normalized_series) <= minimum_context:
        raise ValueError('Series is too short for the requested rolling configuration.')

    model = model_spec.factory()
    step_results: list[RollingForecastStep] = []
    previous_primary_error: float | None = None

    step_index = 0
    train_end_index = int(minimum_context)
    while train_end_index < len(normalized_series):
        context_series = _build_context_slice(normalized_series, train_end_index, config)
        horizon = min(config.forecast_horizon, len(normalized_series) - train_end_index)
        actual = normalized_series[train_end_index:train_end_index + horizon]

        refit_performed = _should_refit_model(
            step_index=step_index,
            previous_error=previous_primary_error,
            config=config,
        )
        if refit_performed:
            model_spec.fit_fn(model, context_series)

        forecast = _truncate_forecast(
            model_spec.predict_fn(model, context_series, horizon),
            horizon,
        )
        metrics = {
            metric_name: compute_metric(metric_name, actual, forecast)
            for metric_name in config.metrics
        }
        metadata = model_spec.metadata_fn(model) if model_spec.metadata_fn else {}
        metadata = {
            **metadata,
            'context_policy': config.context_policy.value,
            'refit_policy': config.refit_policy.value,
        }
        step_results.append(
            RollingForecastStep(
                step_index=step_index,
                train_end_index=train_end_index,
                horizon=horizon,
                metrics=metrics,
                actual_values=tuple(float(value) for value in actual),
                forecast_values=tuple(float(value) for value in forecast),
                refit_performed=refit_performed,
                metadata=metadata,
            )
        )
        previous_primary_error = metrics[config.metrics[0]]
        train_end_index += config.step_size
        step_index += 1

    steps_tuple = tuple(step_results)
    return RollingForecastReport(
        model_name=model_spec.name,
        config=config,
        steps=steps_tuple,
        aggregate_metrics=_aggregate_step_metrics(steps_tuple),
    )


def rolling_report_to_dict(report: RollingForecastReport) -> dict[str, Any]:
    return _make_json_ready({
        'model_name': report.model_name,
        'config': asdict(report.config),
        'steps': [asdict(step) for step in report.steps],
        'aggregate_metrics': dict(report.aggregate_metrics),
    })


def render_rolling_forecast_markdown(report: RollingForecastReport) -> str:
    lines = [
        '# OKHS Rolling Forecast Report',
        '',
        f"- model: {report.model_name}",
        f"- context_policy: {report.config.context_policy.value}",
        f"- refit_policy: {report.config.refit_policy.value}",
        f"- steps: {len(report.steps)}",
        '',
        '## Aggregate Metrics',
    ]
    for metric_name, metric_value in report.aggregate_metrics.items():
        lines.append(f"- {metric_name}: {metric_value:.6f}")

    lines.append('')
    lines.append('## Steps')
    for step in report.steps:
        metric_summary = ', '.join(
            f"{name}={value:.6f}" for name, value in step.metrics.items()
        )
        lines.append(
            f"- step={step.step_index} train_end={step.train_end_index} "
            f"horizon={step.horizon} refit={step.refit_performed} [{metric_summary}]"
        )

    return '\n'.join(lines)


def _z_score(confidence_level: float) -> float:
    return float(NormalDist().inv_cdf(0.5 + confidence_level / 2))


def build_uncertainty_report(
        rolling_report: RollingForecastReport,
        config: UncertaintyConfig,
) -> UncertaintyReport:
    validate_uncertainty_config(config)

    if not rolling_report.steps:
        raise ValueError('Rolling report must contain at least one step.')

    z_score = _z_score(config.confidence_level)
    history_residuals: list[float] = []
    uncertainty_steps: list[UncertaintyStep] = []

    for step in rolling_report.steps:
        step_residuals = [
            actual - forecast
            for actual, forecast in zip(step.actual_values, step.forecast_values)
        ]
        residual_history_for_step = history_residuals.copy()

        flags: list[str] = []
        if len(residual_history_for_step) < config.min_history:
            flags.append('insufficient_history')

        if residual_history_for_step:
            residual_scale = float(
                max(np.std(residual_history_for_step, ddof=0), config.error_floor)
            )
            coverage_proxy = float(
                np.mean(np.abs(residual_history_for_step) <= z_score * residual_scale)
            )
        else:
            residual_scale = float(config.error_floor)
            coverage_proxy = 0.0

        lower = []
        upper = []
        for center in step.forecast_values:
            interval_radius = z_score * residual_scale
            lower.append(float(center - interval_radius))
            upper.append(float(center + interval_radius))
            if interval_radius > config.width_warning_threshold:
                flags.append('wide_interval')

        uncertainty_steps.append(
            UncertaintyStep(
                step_index=step.step_index,
                interval=ForecastInterval(
                    lower=tuple(lower),
                    center=tuple(float(value) for value in step.forecast_values),
                    upper=tuple(upper),
                ),
                residual_scale=residual_scale,
                empirical_coverage_proxy=coverage_proxy,
                quality_flags=tuple(sorted(set(flags))),
            )
        )
        history_residuals.extend(float(value) for value in step_residuals)

    return UncertaintyReport(
        model_name=rolling_report.model_name,
        config=config,
        steps=tuple(uncertainty_steps),
        diagnostics={
            'confidence_level': config.confidence_level,
            'method': config.method.value,
            'history_points': len(history_residuals),
            'z_score': z_score,
        },
    )


def uncertainty_report_to_dict(report: UncertaintyReport) -> dict[str, Any]:
    return _make_json_ready({
        'model_name': report.model_name,
        'config': asdict(report.config),
        'steps': [asdict(step) for step in report.steps],
        'diagnostics': dict(report.diagnostics),
    })


def render_uncertainty_markdown(report: UncertaintyReport) -> str:
    lines = [
        '# OKHS Uncertainty Report',
        '',
        f"- model: {report.model_name}",
        f"- method: {report.config.method.value}",
        f"- confidence_level: {report.config.confidence_level:.2f}",
        '',
        '## Diagnostics',
    ]
    for key, value in report.diagnostics.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.6f}")
        else:
            lines.append(f"- {key}: {value}")

    lines.append('')
    lines.append('## Intervals')
    for step in report.steps:
        lines.append(
            f"- step={step.step_index} residual_scale={step.residual_scale:.6f} "
            f"coverage_proxy={step.empirical_coverage_proxy:.6f} "
            f"flags={','.join(step.quality_flags) if step.quality_flags else 'none'}"
        )

    return '\n'.join(lines)


def _build_q_selection_windows(series: np.ndarray, window_size: int) -> list[np.ndarray]:
    normalized = np.asarray(series, dtype=float).reshape(-1)
    effective_window = max(2, min(window_size, len(normalized)))
    if len(normalized) <= effective_window:
        return [normalized]
    return [
        normalized[index:index + effective_window]
        for index in range(len(normalized) - effective_window + 1)
    ]


def _resolve_q_selection_trace(
        series: np.ndarray,
        q_selection: QSelectionSpec,
        forecast_horizon: int,
        window_size: int,
        n_modes: int,
        method: str | OKHSMethod,
) -> QDecisionTrace:
    str(q_selection.policy)
    normalized_series = np.asarray(series, dtype=float).reshape(-1)

    if q_selection.policy == QPolicy.FIXED or str(q_selection.policy) == QPolicy.FIXED.value:
        return QDecisionTrace(
            policy=QPolicy.FIXED.value,
            selected_q=float(q_selection.fixed_q),
            reason='Fixed q was provided by configuration.',
        )

    if q_selection.policy == QPolicy.DATA_DRIVEN or str(q_selection.policy) == QPolicy.DATA_DRIVEN.value:
        selector = q_selection.selector or DataDrivenQSelector()
        trajectories = _build_q_selection_windows(normalized_series, window_size)
        q_autocorr = float(selector.suggest_q_based_on_autocorrelation(normalized_series))
        q_frequency = float(selector.suggest_q_based_on_frequency(trajectories[:10]))
        selected_q = float(np.mean([q_autocorr, q_frequency]))
        return QDecisionTrace(
            policy=QPolicy.DATA_DRIVEN.value,
            selected_q=selected_q,
            reason='Average of autocorrelation and frequency-based q recommendations.',
            diagnostics={
                'autocorrelation_q': q_autocorr,
                'frequency_q': q_frequency,
                'trajectory_count': len(trajectories),
            },
        )

    if str(q_selection.policy) != 'search':
        raise ValueError(f'Unsupported q-selection policy: {q_selection.policy}')

    if not q_selection.q_grid:
        raise ValueError('Search q-selection requires a non-empty q_grid.')

    validation_size = max(
        forecast_horizon,
        int(round(len(normalized_series) * q_selection.validation_fraction)),
    )
    train_size = len(normalized_series) - validation_size
    if train_size < max(window_size, 2):
        raise ValueError('Search q-selection leaves too few training samples.')

    train_series = normalized_series[:train_size]
    validation_series = normalized_series[train_size:]
    candidate_scores: list[QCandidateScore] = []
    best_q = float(q_selection.q_grid[0])
    best_metric = np.inf

    for candidate_q in q_selection.q_grid:
        candidate_model = OKHSForecaster(
            q=float(candidate_q),
            forecast_horizon=len(validation_series),
            n_modes=n_modes,
            method=method,
            q_policy=QPolicy.FIXED,
        )
        candidate_model.fit(train_series, window_size=window_size)
        forecast = _truncate_forecast(candidate_model.predict(train_series), len(validation_series))
        metric_value = compute_metric(q_selection.selection_metric, validation_series, forecast)
        candidate_scores.append(QCandidateScore(q=float(candidate_q), metric_value=float(metric_value)))
        if metric_value < best_metric:
            best_metric = metric_value
            best_q = float(candidate_q)

    return QDecisionTrace(
        policy='search',
        selected_q=best_q,
        reason=f"Selected q with the best {q_selection.selection_metric} on the internal holdout.",
        candidate_scores=tuple(candidate_scores),
        diagnostics={
            'selection_metric': q_selection.selection_metric,
            'validation_size': validation_size,
        },
    )


def _decision_trace_to_metadata(trace: QDecisionTrace) -> dict[str, Any]:
    payload = asdict(trace)
    payload['candidate_scores'] = [dict(item) for item in payload['candidate_scores']]
    return payload


def build_okhs_forecaster_spec(
        name: str,
        q: float = 0.7,
        forecast_horizon: int = 10,
        n_modes: int = 5,
        method: str | OKHSMethod = OKHSMethod.DMD,
        q_policy: str | QPolicy = QPolicy.FIXED,
        window_size: int = 20,
) -> BenchmarkModelSpec:
    def factory() -> OKHSForecaster:
        return OKHSForecaster(
            q=q,
            forecast_horizon=forecast_horizon,
            n_modes=n_modes,
            method=method,
            q_policy=q_policy,
        )

    def fit_fn(model: OKHSForecaster, series: np.ndarray):
        model.fit(series, window_size=window_size)

    def predict_fn(model: OKHSForecaster, context_series: np.ndarray, horizon: int) -> np.ndarray:
        model.forecast_horizon = horizon
        return np.asarray(model.predict(context_series))

    return BenchmarkModelSpec(
        name=name,
        factory=factory,
        fit_fn=fit_fn,
        predict_fn=predict_fn,
        tags=('okhs', 'forecasting'),
        metadata_fn=lambda model: {
            'policy': str(q_policy),
            'selected_q': float(getattr(model, 'resolved_q_', q)),
            'method': str(method),
        },
    )


def build_okhs_q_orchestrated_spec(
        name: str,
        q_selection: QSelectionSpec,
        forecast_horizon: int = 10,
        n_modes: int = 5,
        method: str | OKHSMethod = OKHSMethod.DMD,
        window_size: int = 20,
) -> BenchmarkModelSpec:
    class QOrchestratedForecaster:
        def __init__(self):
            self.model_: OKHSForecaster | None = None
            self.decision_trace_: QDecisionTrace | None = None

        def fit(self, series: np.ndarray):
            self.decision_trace_ = _resolve_q_selection_trace(
                series=series,
                q_selection=q_selection,
                forecast_horizon=forecast_horizon,
                window_size=window_size,
                n_modes=n_modes,
                method=method,
            )
            self.model_ = OKHSForecaster(
                q=self.decision_trace_.selected_q,
                forecast_horizon=forecast_horizon,
                n_modes=n_modes,
                method=method,
                q_policy=QPolicy.FIXED,
            )
            self.model_.fit(series, window_size=window_size)

        def predict(self, context_series: np.ndarray, horizon: int) -> np.ndarray:
            if self.model_ is None:
                raise ValueError('Q-orchestrated model must be fitted before prediction.')
            self.model_.forecast_horizon = horizon
            return np.asarray(self.model_.predict(context_series))

    def factory() -> QOrchestratedForecaster:
        return QOrchestratedForecaster()

    def fit_fn(model: QOrchestratedForecaster, series: np.ndarray):
        model.fit(series)

    def predict_fn(model: QOrchestratedForecaster, context_series: np.ndarray, horizon: int) -> np.ndarray:
        return model.predict(context_series, horizon)

    def metadata_fn(model: QOrchestratedForecaster) -> dict[str, Any]:
        if model.decision_trace_ is None:
            return {}
        return _decision_trace_to_metadata(model.decision_trace_)

    return BenchmarkModelSpec(
        name=name,
        factory=factory,
        fit_fn=fit_fn,
        predict_fn=predict_fn,
        tags=('okhs', 'forecasting', 'q_orchestrated'),
        metadata_fn=metadata_fn,
    )


def build_naive_last_value_spec(name: str = 'naive_last_value') -> BenchmarkModelSpec:
    class NaiveLastValueModel:
        def __init__(self):
            self.last_value_ = 0.0

        def fit(self, series: np.ndarray):
            self.last_value_ = float(np.asarray(series)[-1])

        def predict(self, horizon: int) -> np.ndarray:
            return np.full(horizon, self.last_value_, dtype=float)

    def factory() -> NaiveLastValueModel:
        return NaiveLastValueModel()

    def fit_fn(model: NaiveLastValueModel, series: np.ndarray):
        model.fit(series)

    def predict_fn(model: NaiveLastValueModel, context_series: np.ndarray, horizon: int) -> np.ndarray:
        del context_series
        return model.predict(horizon)

    return BenchmarkModelSpec(
        name=name,
        factory=factory,
        fit_fn=fit_fn,
        predict_fn=predict_fn,
        tags=('baseline', 'forecasting'),
    )


def build_naive_mean_spec(name: str = 'naive_mean') -> BenchmarkModelSpec:
    class NaiveMeanModel:
        def __init__(self):
            self.mean_value_ = 0.0

        def fit(self, series: np.ndarray):
            self.mean_value_ = float(np.mean(np.asarray(series, dtype=float)))

        def predict(self, horizon: int) -> np.ndarray:
            return np.full(horizon, self.mean_value_, dtype=float)

    def factory() -> NaiveMeanModel:
        return NaiveMeanModel()

    def fit_fn(model: NaiveMeanModel, series: np.ndarray):
        model.fit(series)

    def predict_fn(model: NaiveMeanModel, context_series: np.ndarray, horizon: int) -> np.ndarray:
        del context_series
        return model.predict(horizon)

    return BenchmarkModelSpec(
        name=name,
        factory=factory,
        fit_fn=fit_fn,
        predict_fn=predict_fn,
        tags=('baseline', 'forecasting'),
    )


def build_naive_drift_spec(name: str = 'naive_drift') -> BenchmarkModelSpec:
    class NaiveDriftModel:
        def __init__(self):
            self.start_value_ = 0.0
            self.end_value_ = 0.0
            self.length_ = 1

        def fit(self, series: np.ndarray):
            normalized = np.asarray(series, dtype=float).reshape(-1)
            self.start_value_ = float(normalized[0])
            self.end_value_ = float(normalized[-1])
            self.length_ = len(normalized)

        def predict(self, horizon: int) -> np.ndarray:
            if self.length_ <= 1:
                return np.full(horizon, self.end_value_, dtype=float)

            slope = (self.end_value_ - self.start_value_) / (self.length_ - 1)
            steps = np.arange(1, horizon + 1, dtype=float)
            return self.end_value_ + slope * steps

    def factory() -> NaiveDriftModel:
        return NaiveDriftModel()

    def fit_fn(model: NaiveDriftModel, series: np.ndarray):
        model.fit(series)

    def predict_fn(model: NaiveDriftModel, context_series: np.ndarray, horizon: int) -> np.ndarray:
        del context_series
        return model.predict(horizon)

    return BenchmarkModelSpec(
        name=name,
        factory=factory,
        fit_fn=fit_fn,
        predict_fn=predict_fn,
        tags=('baseline', 'forecasting'),
    )
