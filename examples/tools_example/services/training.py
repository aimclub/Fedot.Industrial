from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.industrial import (
    ModelSpec,
    run_forecasting_benchmark_suite,
    run_tsc_benchmark_suite,
    run_tser_benchmark_suite,
)
from benchmark.industrial.core import TaskType
from benchmark.industrial.experiments.presets import (
    build_local_m4_suite_config,
    build_local_tser_suite_config,
    build_local_ucr_suite_config,
)

from examples.tools_example.contracts import ToolArtifact, ToolRequest, ToolResponse
from examples.tools_example.services.common import config_summary, result_summary


def train_model(request: ToolRequest) -> ToolResponse:
    config = build_training_config(request.payload, persist_on_run=not request.dry_run)
    if request.dry_run:
        return ToolResponse(
            name=request.name,
            status="dry_run",
            dry_run=True,
            message="Training config built; execution was not requested.",
            data={"config": config_summary(config)},
        )

    result = _run_config(config)
    artifacts = tuple(
        ToolArtifact(kind=record.kind, path=record.path, format=record.format)
        for record in result.artifact_manifest
    )
    return ToolResponse(
        name=request.name,
        status="success",
        dry_run=False,
        message="Industrial benchmark training completed.",
        data={"result": result_summary(result), "config": config_summary(config)},
        artifacts=artifacts,
    )


def build_training_config(payload: dict[str, Any], *, persist_on_run: bool):
    task_type = TaskType(str(payload.get("task_type", TaskType.TS_CLASSIFICATION.value)))
    dataset_name = payload.get("dataset_name")
    subset = payload.get("subset")
    output_dir = payload.get("output_dir")
    models = _model_specs(payload.get("models"))

    if task_type is TaskType.TS_CLASSIFICATION:
        return build_local_ucr_suite_config(
            dataset_name=str(dataset_name or "Lightning7"),
            output_dir=Path(output_dir) if output_dir else None,
            persist_on_run=persist_on_run,
            models=models or None,
        )
    if task_type is TaskType.TS_REGRESSION:
        return build_local_tser_suite_config(
            dataset_name=str(dataset_name or "NaturalGasPricesSentiment"),
            output_dir=Path(output_dir) if output_dir else None,
            persist_on_run=persist_on_run,
            models=models or None,
        )
    if task_type is TaskType.FORECASTING:
        sample_size = payload.get("sample_size", 3)
        return build_local_m4_suite_config(
            subset=str(subset or "daily"),
            sample_size=int(sample_size) if sample_size is not None else None,
            output_dir=Path(output_dir) if output_dir else None,
            persist_on_run=persist_on_run,
            models=models or None,
            include_optional_external=bool(payload.get("include_optional_external", False)),
        )
    raise ValueError(f"Unsupported training task type: {task_type}")


def _run_config(config):
    if config.task_type is TaskType.TS_CLASSIFICATION:
        return run_tsc_benchmark_suite(config)
    if config.task_type is TaskType.TS_REGRESSION:
        return run_tser_benchmark_suite(config)
    if config.task_type is TaskType.FORECASTING:
        return run_forecasting_benchmark_suite(config)
    raise ValueError(f"Unsupported config task type: {config.task_type}")


def _model_specs(payload: Any) -> tuple[ModelSpec, ...]:
    if not payload:
        return ()
    return tuple(ModelSpec(**dict(item)) for item in payload)


__all__ = ["build_training_config", "train_model"]
