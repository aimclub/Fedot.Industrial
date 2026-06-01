from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.models.detection.progress_policy import (
    DetectionProgressPolicy,
    resolve_detection_progress_policy,
)
from fedot_ind.core.models.detection.runtime import (
    DetectionSplitKind,
    DetectionSplitSpec,
    ensure_detection_array,
)
# from fedot_ind.core.metrics.metrics_implementation import (
#     DETECTION_METRICS_TO_MINIMIZE,
#     calculate_detection_metric,
# )

from fedot_ind.core.repository.constanst_repository import FEDOT_GET_METRICS
from fedot_ind.core.metrics.metric_library import METRIC_REGISTRY, METRICS_TO_MINIMIZE

SUPPORTED_ANOMALY_DETECTION_METRICS = tuple(METRIC_REGISTRY['anomaly_detection'])
DETECTION_METRICS_TO_MINIMIZE = tuple(set(METRICS_TO_MINIMIZE) & set(SUPPORTED_ANOMALY_DETECTION_METRICS))

from fedot_ind.core.models.detection.stage_tuning import build_detection_stage_tuning_plan
from fedot_ind.core.operation.interfaces.detection_runtime_strategy import DETECTION_RUNTIME_MODELS
from fedot_ind.core.repository.detection_registry import canonical_detection_model_name, detection_family_for

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


@dataclass(frozen=True)
class DetectionSeriesEvaluation:
    """Single-series evaluation snapshot for one detector parameter set.

    This structure is the atomic unit used in stage-tuning reports:
    model/params in, metric + labels out.
    """
    model_name: str
    canonical_model_name: str
    family: str
    parameters: dict[str, Any]
    metric_name: str
    metric_value: float
    labels: tuple[int, ...]
    target: tuple[int, ...]
    split_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'model_name': self.model_name,
            'canonical_model_name': self.canonical_model_name,
            'family': self.family,
            'parameters': dict(self.parameters),
            'metric_name': self.metric_name,
            'metric_value': float(self.metric_value),
            'labels': list(self.labels),
            'target': list(self.target),
            'split_metadata': dict(self.split_metadata),
            **self.metadata,
        }


@dataclass(frozen=True)
class DetectionSequentialStageTuningResult:
    """Sequential tuning trace across stage groups defined in tuning plan."""
    model_name: str
    canonical_model_name: str
    family: str
    base_parameters: dict[str, Any]
    best_parameters: dict[str, Any]
    best_score: float
    stage_history: tuple[dict[str, Any], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'model_name': self.model_name,
            'canonical_model_name': self.canonical_model_name,
            'family': self.family,
            'base_parameters': dict(self.base_parameters),
            'best_parameters': dict(self.best_parameters),
            'best_score': float(self.best_score),
            'stage_history': list(self.stage_history),
            **self.metadata,
        }


@dataclass(frozen=True)
class DetectionSeriesStageTuningResult:
    """Final tuning report containing baseline and best-parameter evaluations."""
    model_name: str
    canonical_model_name: str
    family: str
    sequential_result: DetectionSequentialStageTuningResult
    baseline_evaluation: DetectionSeriesEvaluation
    best_evaluation: DetectionSeriesEvaluation
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


def _normalize_base_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Drop ``None`` values to avoid overriding detector defaults with nulls."""
    return {key: value for key, value in dict(params or {}).items() if value is not None}


def _split_series(
        values: np.ndarray,
        labels: np.ndarray,
        split_spec: DetectionSplitSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build train/calibration split from a labeled time series.

    Returns
    -------
    tuple
        ``(train_values, train_labels, calibration_values, calibration_labels)``.
        If calibration window is empty, it falls back to train slice.
    """
    n_samples = int(values.shape[0])
    train_end = max(1, int(round(n_samples * split_spec.train_fraction)))
    remaining = max(0, n_samples - train_end)
    calibration_size = int(round(remaining * split_spec.calibration_fraction))
    calibration_end = min(n_samples, train_end + calibration_size)
    train_values = values[:train_end]
    train_labels = labels[:train_end]
    calibration_values = values[train_end:calibration_end]
    calibration_labels = labels[train_end:calibration_end]
    if calibration_values.size == 0:
        calibration_values = train_values
        calibration_labels = train_labels
    return train_values, train_labels, calibration_values, calibration_labels


def _build_detector(model_name: str, params: dict[str, Any]):
    """Instantiate runtime detector by canonical name from registry map."""
    canonical_name = canonical_detection_model_name(model_name)
    if canonical_name not in DETECTION_RUNTIME_MODELS:
        raise ValueError(f'Unsupported detection model for stage tuning: {model_name}')
    return DETECTION_RUNTIME_MODELS[canonical_name](OperationParameters(**params))


def _evaluate_parameters(
        model_name: str,
        *,
        values: np.ndarray,
        labels: np.ndarray,
        parameters: dict[str, Any],
        metric_name: str,
        split_spec: DetectionSplitSpec,
) -> DetectionSeriesEvaluation:
    """Fit/evaluate one detector configuration on one series split."""
    canonical_name = canonical_detection_model_name(model_name)
    series = ensure_detection_array(values)
    target = np.asarray(labels, dtype=int).reshape(-1)
    train_values, _, calibration_values, calibration_labels = _split_series(series, target, split_spec)
    detector = _build_detector(model_name, parameters)
    # detector.fit(InputData(features=train_values))
    detector.fit(train_values)
    score_series = detector.score_series_on_values(calibration_values)
    predicted = np.asarray(score_series.labels, dtype=int).reshape(-1)
    # metric_value = _compute_detection_metric(metric_name, calibration_labels, predicted)
    
    # metric_values = calculate_detection_metric(
    #     target=calibration_labels,
    #     labels=predicted,
    #     metric_names=(metric_name,),
    # )

    metric_values = FEDOT_GET_METRICS['anomaly_detection'](target=calibration_labels,
                                                           predicted_labels=predicted,
                                                           predicted_probs=None, # Можно добавить, для некоторых метрик будет полезно
                                                           metric_names=tuple(metric_name,),
                                                        #    rounding_order=4,   # Можно добавить
                                                           return_dataframe = False)
    metric_value = float(metric_values[metric_name])
    return DetectionSeriesEvaluation(
        model_name=model_name,
        canonical_model_name=canonical_name,
        family=detection_family_for(canonical_name),
        parameters=dict(parameters),
        metric_name=metric_name,
        metric_value=float(metric_value),
        labels=tuple(int(value) for value in predicted.tolist()),
        target=tuple(int(value) for value in calibration_labels.tolist()),
        split_metadata={
            'train_size': int(train_values.shape[0]),
            'calibration_size': int(calibration_values.shape[0]),
            'split_kind': split_spec.kind.value,
        },
    )


def _candidate_values(parameter_name: str, current_value: Any, max_values: int) -> tuple[Any, ...]:
    """Generate compact candidate values around current parameter value."""
    if isinstance(current_value, bool):
        return (current_value,)
    if isinstance(current_value, int):
        deltas = (-max(1, current_value // 4), 0, max(1, current_value // 4))
        candidates = tuple(dict.fromkeys(max(1, current_value + delta) for delta in deltas))[:max_values]
        return candidates or (current_value,)
    if isinstance(current_value, float):
        scale = max(abs(current_value) * 0.25, 1e-3)
        return tuple(dict.fromkeys((current_value - scale, current_value, current_value + scale)))[:max_values]
    if isinstance(current_value, str):
        return (current_value,)
    return (current_value,)


def _stage_candidate_grid(
        group_parameters: tuple[str, ...],
        current_parameters: dict[str, Any],
        *,
        max_values_per_parameter: int,
        max_stage_candidates: int,
) -> tuple[dict[str, Any], ...]:
    """Construct Cartesian candidate grid for one tuning stage."""
    parameter_values: list[tuple[str, tuple[Any, ...]]] = []
    for parameter_name in group_parameters:
        if parameter_name not in current_parameters:
            continue
        values = _candidate_values(
            parameter_name,
            current_parameters[parameter_name],
            max_values_per_parameter,
        )
        parameter_values.append((parameter_name, values))
    if not parameter_values:
        return ({},)
    grid: list[dict[str, Any]] = []
    for combination in product(*(values for _, values in parameter_values)):
        candidate = {
            parameter_name: value
            for (parameter_name, _), value in zip(parameter_values, combination)
        }
        grid.append(candidate)
        if len(grid) >= max_stage_candidates:
            break
    return tuple(grid[:max_stage_candidates])


def _is_better_detection_score(candidate_score: float, best_score: float, metric_name: str) -> bool:
    """Сравнение значения метрик учитывая направление, 
    указанное в реестре общих метрик."""
    if metric_name in DETECTION_METRICS_TO_MINIMIZE:
        return candidate_score <= best_score
    return candidate_score >= best_score


def run_detection_stage_tuning_on_series(
        model_name: str,
        *,
        values: np.ndarray,
        labels: np.ndarray,
        base_params: dict[str, Any] | None = None,
        stage_updates: dict[str, Any] | None = None,
        metric_name: str,
        split_spec: DetectionSplitSpec | None = None,
        max_values_per_parameter: int = 3,
        max_stage_candidates: int = 16,
        progress_policy: DetectionProgressPolicy | dict[str, Any] | bool | None = None,
) -> DetectionSeriesStageTuningResult:
    """Run sequential stage tuning for one detector on one labeled series.

    Parameters
    ----------
    model_name:
        Adapter/detector name (canonical or alias).
    values:
        Input time-series values, univariate or multivariate.
    labels:
        Point-wise anomaly labels aligned with ``values``.
    base_params:
        Initial detector parameters before stage-wise updates.
    stage_updates:
        Reserved for future explicit per-stage updates (currently ignored).
    metric_name:
        Metric optimized during tuning (`accuracy`, `balanced_accuracy`, `f1_macro`).
    split_spec:
        Optional split configuration; defaults to temporal split.
    max_values_per_parameter:
        Upper bound on generated candidate values per parameter.
    max_stage_candidates:
        Upper bound on evaluated combinations per stage.
    progress_policy:
        Optional progress display policy.
    """
    del stage_updates
    resolved_params = _normalize_base_params(base_params)
    resolved_split = split_spec or DetectionSplitSpec(kind=DetectionSplitKind.TEMPORAL)
    resolved_progress = resolve_detection_progress_policy(progress_policy)
    plan = build_detection_stage_tuning_plan(model_name, resolved_params)
    if not plan.groups:
        baseline = _evaluate_parameters(
            model_name,
            values=values,
            labels=labels,
            parameters=resolved_params,
            metric_name=metric_name,
            split_spec=resolved_split,
        )
        sequential = DetectionSequentialStageTuningResult(
            model_name=model_name,
            canonical_model_name=plan.canonical_model_name,
            family=plan.family,
            base_parameters=resolved_params,
            best_parameters=resolved_params,
            best_score=baseline.metric_value,
            stage_history=(),
            metadata={'supports_stage_tuning': False},
        )
        return DetectionSeriesStageTuningResult(
            model_name=model_name,
            canonical_model_name=plan.canonical_model_name,
            family=plan.family,
            sequential_result=sequential,
            baseline_evaluation=baseline,
            best_evaluation=baseline,
            metadata={'improved': False, 'baseline_score': baseline.metric_value, 'best_score': baseline.metric_value},
        )

    current_parameters = dict(resolved_params)
    stage_history: list[dict[str, Any]] = []
    baseline_evaluation = _evaluate_parameters(
        model_name,
        values=values,
        labels=labels,
        parameters=current_parameters,
        metric_name=metric_name,
        split_spec=resolved_split,
    )
    best_score = baseline_evaluation.metric_value
    best_parameters = dict(current_parameters)

    stage_iterator = plan.groups
    if resolved_progress.is_stage_tuning_enabled():
        stage_iterator = tqdm(plan.groups, **resolved_progress.tqdm_kwargs(
            scope='stage_tuning',
            desc='detection stage tuning',
            unit='stage',
        ))

    for group in stage_iterator:
        evaluations: list[dict[str, Any]] = []
        stage_best_score = best_score
        stage_best_parameters = dict(current_parameters)
        # Stage-local search: vary only parameters assigned to current stage.
        for candidate in _stage_candidate_grid(
                group.parameters,
                current_parameters,
                max_values_per_parameter=max_values_per_parameter,
                max_stage_candidates=max_stage_candidates,
        ):
            candidate_parameters = {**current_parameters, **candidate}
            evaluation = _evaluate_parameters(
                model_name,
                values=values,
                labels=labels,
                parameters=candidate_parameters,
                metric_name=metric_name,
                split_spec=resolved_split,
            )
            evaluations.append({
                'parameters': dict(candidate),
                'score': float(evaluation.metric_value),
            })
            # if evaluation.metric_value >= stage_best_score:
            if _is_better_detection_score(evaluation.metric_value, stage_best_score, metric_name):
                stage_best_score = evaluation.metric_value
                stage_best_parameters = dict(candidate_parameters)
        # Sequential tuning semantics: next stage starts from best params of previous stage.
        current_parameters = stage_best_parameters
        # if stage_best_score >= best_score:
        if _is_better_detection_score(stage_best_score, best_score, metric_name):
            best_score = stage_best_score
            best_parameters = dict(current_parameters)
        stage_history.append({
            'stage': group.stage,
            'best_parameters': dict(stage_best_parameters),
            'best_score': float(stage_best_score),
            'evaluations': evaluations,
        })

    best_evaluation = _evaluate_parameters(
        model_name,
        values=values,
        labels=labels,
        parameters=best_parameters,
        metric_name=metric_name,
        split_spec=resolved_split,
    )
    sequential_result = DetectionSequentialStageTuningResult(
        model_name=model_name,
        canonical_model_name=plan.canonical_model_name,
        family=plan.family,
        base_parameters=resolved_params,
        best_parameters=best_parameters,
        best_score=float(best_score),
        stage_history=tuple(stage_history),
        metadata={'supports_stage_tuning': True, 'progress_policy': resolved_progress.to_dict()},
    )
    return DetectionSeriesStageTuningResult(
        model_name=model_name,
        canonical_model_name=plan.canonical_model_name,
        family=plan.family,
        sequential_result=sequential_result,
        baseline_evaluation=baseline_evaluation,
        best_evaluation=best_evaluation,
        metadata={
            'improved': bool(_is_better_detection_score(best_score, baseline_evaluation.metric_value, metric_name)),
            'baseline_score': float(baseline_evaluation.metric_value),
            'best_score': float(best_score),
        },
    )
