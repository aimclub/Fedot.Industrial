"""Compatibility entrypoints with the Industrial parameter catalogue."""

from typing import Any, Callable

from industrial_forecasting.progress import ForecastingProgressPolicy
from industrial_forecasting.tuning.execution import (
    ForecastingSequentialStageTuningResult, ForecastingStageTuningExecution,
    SequentialStageTuningRunner as _SequentialStageTuningRunner,
    StageCandidateEvaluation, StageTuningExecutionStep,
    build_forecasting_stage_tuning_execution as _build_execution,
)

from .stage_tuning import _operation_parameter_space

__all__ = [
    'ForecastingSequentialStageTuningResult', 'ForecastingStageTuningExecution', 'SequentialStageTuningRunner',
    'StageCandidateEvaluation', 'StageTuningExecutionStep',
    'build_forecasting_stage_tuning_execution', 'run_sequential_stage_tuning',
]


def build_forecasting_stage_tuning_execution(
        model_name: str, *, base_params: dict[str, Any] | None = None, stage_updates: dict[str, Any] | None = None,
) -> ForecastingStageTuningExecution:
    return _build_execution(
        model_name, base_params=base_params, stage_updates=stage_updates,
        parameter_space=_operation_parameter_space(model_name),
    )


class SequentialStageTuningRunner(_SequentialStageTuningRunner):
    def __post_init__(self):
        if self.parameter_space is None:
            self.parameter_space = _operation_parameter_space(self.model_name)
        super().__post_init__()


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
    return SequentialStageTuningRunner(
        model_name=model_name, objective=objective, base_params=base_params, stage_updates=stage_updates,
        max_values_per_parameter=max_values_per_parameter, max_stage_candidates=max_stage_candidates,
        show_progress=show_progress, progress_policy=progress_policy,
    ).run()
