from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from benchmark.industrial.core import BenchmarkSuiteConfig, to_plain_data
from examples.utils.config_io import load_versioned_json

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXAMPLE_ROOT.parents[1]
DEFAULTS_PATH = EXAMPLE_ROOT / "tool_defaults.json"
DEFAULTS_VERSION = "industrial_tools_examples@1"


@lru_cache(maxsize=1)
def load_tool_defaults(path: str | Path = DEFAULTS_PATH) -> dict[str, Any]:
    return load_versioned_json(
        path,
        expected_version=DEFAULTS_VERSION,
        description="tools example defaults",
    )


def resolve_project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def config_summary(config: BenchmarkSuiteConfig) -> dict[str, Any]:
    return {
        "task_type": config.task_type.value,
        "datasets": [to_plain_data(dataset) for dataset in config.datasets],
        "models": [to_plain_data(model) for model in config.models],
        "metrics": list(config.metrics),
        "artifact_spec": to_plain_data(config.artifact_spec),
        "run_spec": to_plain_data(config.run_spec),
    }


def result_summary(result) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "task_type": result.config.task_type.value,
        "successful_runs": sum(1 for record in result.run_records if record.status.value == "success"),
        "failed_runs": sum(1 for record in result.run_records if record.status.value == "failed"),
        "primary_metric": result.aggregate_report.primary_metric,
        "status_counts": dict(result.aggregate_report.status_counts),
        "leaderboard_rows": list(result.aggregate_report.leaderboard_rows),
    }


__all__ = [
    "EXAMPLE_ROOT",
    "PROJECT_ROOT",
    "config_summary",
    "load_tool_defaults",
    "resolve_project_path",
    "result_summary",
]
