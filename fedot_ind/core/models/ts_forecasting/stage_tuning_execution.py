from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Any, Callable

try:  # pragma: no cover - progress bar is optional in lightweight envs
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

from .progress_policy import ForecastingProgressPolicy, resolve_forecasting_progress_policy
from .stage_tuning import (
    ForecastingStageSearchSpace,
    build_forecasting_stage_search_spaces,
    build_forecasting_stage_tuning_plan,
)


@dataclass(frozen=True)
class StageTuningExecutionStep:
    stage: str
    depends_on: tuple[str, ...]
    allowed_parameters: tuple[str, ...]
    search_space_parameters: tuple[str, ...]
    proposed_parameters: dict[str, Any]
    applied_parameters: dict[str, Any]
    ignored_parameters: dict[str, Any]
    resulting_parameters: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForecastingStageTuningExecution:
    model_name: str
    canonical_model_name: str
    family: str
    base_parameters: dict[str, Any]
    final_parameters: dict[str, Any]
    steps: tuple[StageTuningExecutionStep, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'model_name': self.model_name,
            'canonical_model_name': self.canonical_model_name,
            'family': self.family,
            'base_parameters': dict(self.base_parameters),
            'final_parameters': dict(self.final_parameters),
            'steps': [step.to_dict() for step in self.steps],
            **self.metadata,
        }


@dataclass(frozen=True)
class StageCandidateEvaluation:
    stage: str
    parameters: dict[str, Any]
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForecastingSequentialStageTuningResult:
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


def _normalize_stage_updates(stage_updates: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for stage_name, value in dict(stage_updates or {}).items():
        normalized[str(stage_name)] = dict(value or {})
    return normalized


def _representative_values_from_spec(spec: dict[str, Any], max_values_per_parameter: int) -> tuple[Any, ...]:
    scope = spec.get('sampling-scope')
    parameter_type = spec.get('type')
    if scope is None:
        return ()

    if isinstance(scope, list) and len(scope) == 1 and isinstance(scope[0], list):
        values = tuple(scope[0][:max_values_per_parameter])
        return values

    if isinstance(scope, list) and len(scope) == 2 and all(isinstance(item, (int, float)) for item in scope):
        low, high = scope
        if parameter_type == 'continuous':
            mid = (float(low) + float(high)) / 2.0
            values = (float(low), float(mid), float(high))
        else:
            low_i = int(round(low))
            high_i = int(round(high))
            mid_i = int(round((low_i + high_i) / 2))
            values = (low_i, mid_i, high_i)
        return tuple(dict.fromkeys(values))[:max_values_per_parameter]

    if isinstance(scope, list):
        return tuple(scope[:max_values_per_parameter])
    return ()


def _build_stage_candidate_grid(
        search_space: ForecastingStageSearchSpace,
        *,
        current_parameters: dict[str, Any],
        proposed_parameters: dict[str, Any] | None = None,
        max_values_per_parameter: int = 3,
        max_stage_candidates: int = 16,
) -> tuple[dict[str, Any], ...]:
    parameter_values: list[tuple[str, tuple[Any, ...]]] = []
    for parameter_name, spec in search_space.parameter_space.items():
        values = _representative_values_from_spec(spec, max_values_per_parameter)
        if not values and parameter_name in current_parameters:
            values = (current_parameters[parameter_name],)
        elif parameter_name in current_parameters:
            values = tuple(dict.fromkeys((current_parameters[parameter_name], *values)))
        if values:
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
    preferred_candidates: list[dict[str, Any]] = []
    if proposed_parameters:
        preferred_candidates.append(dict(proposed_parameters))
    preferred_candidates.append({})

    for candidate in reversed(preferred_candidates):
        if candidate in grid:
            grid.remove(candidate)
        grid.insert(0, candidate)
    return tuple(grid[:max_stage_candidates])


def _execute_stage_group(
        *,
        current_params: dict[str, Any],
        group,
        search_space: ForecastingStageSearchSpace | None,
        proposed_parameters: dict[str, Any],
) -> tuple[dict[str, Any], StageTuningExecutionStep]:
    allowed = tuple(group.parameters)
    allowed_set = set(allowed)
    applied = {key: value for key, value in proposed_parameters.items() if key in allowed_set}
    ignored = {key: value for key, value in proposed_parameters.items() if key not in allowed_set}
    updated_params = {**current_params, **applied}
    step = StageTuningExecutionStep(
        stage=group.stage,
        depends_on=tuple(group.depends_on),
        allowed_parameters=allowed,
        search_space_parameters=tuple((search_space.parameter_space or {}).keys()) if search_space else (),
        proposed_parameters=dict(proposed_parameters),
        applied_parameters=applied,
        ignored_parameters=ignored,
        resulting_parameters=dict(updated_params),
        metadata={} if search_space is None else dict(search_space.metadata),
    )
    return updated_params, step


def build_forecasting_stage_tuning_execution(
        model_name: str,
        *,
        base_params: dict[str, Any] | None = None,
        stage_updates: dict[str, Any] | None = None,
) -> ForecastingStageTuningExecution:
    resolved_base_params = dict(base_params or {})
    plan = build_forecasting_stage_tuning_plan(model_name, params=resolved_base_params)
    search_spaces = {
        space.stage: space
        for space in build_forecasting_stage_search_spaces(model_name, params=resolved_base_params)
    }
    normalized_updates = _normalize_stage_updates(stage_updates)

    current_params = dict(resolved_base_params)
    steps: list[StageTuningExecutionStep] = []
    for group in plan.groups:
        proposed_parameters = normalized_updates.get(group.stage, {})
        current_params, step = _execute_stage_group(
            current_params=current_params,
            group=group,
            search_space=search_spaces.get(group.stage),
            proposed_parameters=proposed_parameters,
        )
        steps.append(step)

    return ForecastingStageTuningExecution(
        model_name=model_name,
        canonical_model_name=plan.canonical_model_name,
        family=plan.family,
        base_parameters=resolved_base_params,
        final_parameters=current_params,
        steps=tuple(steps),
        metadata={
            'supports_simultaneous_tuning': plan.metadata.get('supports_simultaneous_tuning', False),
            'stage_count': len(steps),
        },
    )


def run_sequential_stage_tuning(
        model_name: str,
        *,
        objective: Callable[[dict[str, Any]], float],
        base_params: dict[str, Any] | None = None,
        stage_updates: dict[str, Any] | None = None,
        max_values_per_parameter: int = 3,
        max_stage_candidates: int = 16,
        show_progress: bool | None = None,
        progress_policy: ForecastingProgressPolicy | dict[str, Any] | bool | None = None,
) -> ForecastingSequentialStageTuningResult:
    resolved_progress_policy = resolve_forecasting_progress_policy(
        progress_policy,
        show_progress=show_progress,
    )
    execution = build_forecasting_stage_tuning_execution(
        model_name,
        base_params=base_params,
        stage_updates=stage_updates,
    )
    search_spaces = {
        space.stage: space
        for space in build_forecasting_stage_search_spaces(model_name, params=execution.final_parameters)
    }

    current_best_params = dict(execution.base_parameters)
    current_best_score = float(objective(current_best_params))
    stage_history: list[dict[str, Any]] = []

    stage_iterator = tqdm(
        execution.steps,
        **resolved_progress_policy.tqdm_kwargs(
            scope='stage_tuning',
            desc=f'Stage tuning: {execution.canonical_model_name}',
            unit='stage',
        ),
    )
    for step in stage_iterator:
        stage_search_space = search_spaces.get(step.stage)
        candidate_grid = _build_stage_candidate_grid(
            stage_search_space or ForecastingStageSearchSpace(
                model_name=execution.model_name,
                canonical_model_name=execution.canonical_model_name,
                family=execution.family,
                stage=step.stage,
                parameter_space={},
                depends_on=step.depends_on,
            ),
            current_parameters=current_best_params,
            proposed_parameters=step.proposed_parameters,
            max_values_per_parameter=max_values_per_parameter,
            max_stage_candidates=max_stage_candidates,
        )

        evaluations: list[StageCandidateEvaluation] = []
        best_stage_params = dict(current_best_params)
        best_stage_score = current_best_score
        candidate_iterator = tqdm(
            candidate_grid,
            **resolved_progress_policy.tqdm_kwargs(
                scope='stage_tuning',
                desc=f'  {step.stage}',
                unit='candidate',
            ),
        )
        for candidate in candidate_iterator:
            merged_params, applied_step = _execute_stage_group(
                current_params=current_best_params,
                group=type('StageGroup', (), {
                    'stage': step.stage,
                    'parameters': step.allowed_parameters,
                    'depends_on': step.depends_on,
                })(),
                search_space=stage_search_space,
                proposed_parameters=candidate,
            )
            score = float(objective(merged_params))
            evaluations.append(
                StageCandidateEvaluation(
                    stage=step.stage,
                    parameters=dict(candidate),
                    score=score,
                    metadata={
                        'applied_parameters': applied_step.applied_parameters,
                        'ignored_parameters': applied_step.ignored_parameters,
                    },
                )
            )
            if score < best_stage_score:
                best_stage_score = score
                best_stage_params = merged_params
            if resolved_progress_policy.show_postfix and hasattr(candidate_iterator, 'set_postfix'):
                candidate_iterator.set_postfix(best=f'{float(best_stage_score):.5f}', current=f'{float(score):.5f}')

        current_best_params = dict(best_stage_params)
        current_best_score = float(best_stage_score)
        if resolved_progress_policy.show_postfix and hasattr(stage_iterator, 'set_postfix'):
            stage_iterator.set_postfix(best=f'{float(current_best_score):.5f}', stage=step.stage)
        stage_history.append(
            {
                'stage': step.stage,
                'depends_on': step.depends_on,
                'candidate_count': len(candidate_grid),
                'best_score': float(best_stage_score),
                'best_parameters': dict(current_best_params),
                'evaluations': [evaluation.to_dict() for evaluation in evaluations],
            }
        )

    return ForecastingSequentialStageTuningResult(
        model_name=execution.model_name,
        canonical_model_name=execution.canonical_model_name,
        family=execution.family,
        base_parameters=execution.base_parameters,
        best_parameters=current_best_params,
        best_score=float(current_best_score),
        stage_history=tuple(stage_history),
        metadata={
            'supports_simultaneous_tuning': execution.metadata.get('supports_simultaneous_tuning', False),
            'stage_count': len(stage_history),
            'max_values_per_parameter': int(max_values_per_parameter),
            'max_stage_candidates': int(max_stage_candidates),
            'progress_policy': resolved_progress_policy.to_dict(),
        },
    )
