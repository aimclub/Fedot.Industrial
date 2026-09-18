from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.experiments.kernel_learning.configs import KernelLearningTwoStageUCRExperimentConfig
from benchmark.industrial.core import to_plain_data

from examples.tools_example.contracts import ToolRequest, ToolResponse


def run_evolution_optimization(request: ToolRequest) -> ToolResponse:
    config = _build_two_stage_config(request.payload)
    preview = {
        "data_root": str(config.data_root),
        "datasets": list(config.datasets),
        "stage1_output_dir": str(config.stage1_output_dir),
        "stage2_output_dir": str(config.stage2_output_dir),
        "stage1_run_id": config.stage1_run_id,
        "run_stage1": config.run_stage1,
        "generator_names": list(config.generator_names),
        "metrics": list(config.metrics),
        "timeout_minutes": config.timeout_minutes,
        "pop_size": config.pop_size,
    }
    if request.dry_run:
        return ToolResponse(
            name=request.name,
            status="dry_run",
            dry_run=True,
            message="Evolution optimization config built; execution was not requested.",
            data={"config": preview},
        )

    stage1_result = config.load_or_run_stage1()
    stage2_result = config.run_stage2(stage1_result)
    return ToolResponse(
        name=request.name,
        status="success",
        dry_run=False,
        message="Kernel Learning two-stage evolutionary optimization completed.",
        data={"config": preview, "stage2_result": to_plain_data(stage2_result)},
    )


def _build_two_stage_config(payload: dict[str, Any]) -> KernelLearningTwoStageUCRExperimentConfig:
    datasets = tuple(str(item) for item in payload.get("datasets", ()) if str(item))
    return KernelLearningTwoStageUCRExperimentConfig(
        datasets=datasets, stage1_output_dir=Path(payload["stage1_output_dir"])
        if payload.get("stage1_output_dir") else KernelLearningTwoStageUCRExperimentConfig.stage1_output_dir,
        stage2_output_dir=Path(payload["stage2_output_dir"])
        if payload.get("stage2_output_dir") else KernelLearningTwoStageUCRExperimentConfig.stage2_output_dir,
        stage1_run_id=payload.get("stage1_run_id", KernelLearningTwoStageUCRExperimentConfig.stage1_run_id),
        run_stage1=bool(payload.get("run_stage1", False)),
        timeout_minutes=int(
            payload.get("timeout_minutes", KernelLearningTwoStageUCRExperimentConfig.timeout_minutes)),
        pop_size=int(payload.get("pop_size", KernelLearningTwoStageUCRExperimentConfig.pop_size)),)


__all__ = ["run_evolution_optimization"]
