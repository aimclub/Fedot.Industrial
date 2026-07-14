from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from benchmark.industrial import (
    ArtifactSpec,
    BenchmarkSuiteConfig,
    DatasetSpec,
    ModelSpec,
    RunSpec,
    TaskType,
    run_tsc_benchmark_suite,
    run_tser_benchmark_suite,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = Path(__file__).with_name("example_defaults.json")
DEFAULTS_VERSION = "industrial_current_api_examples@1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "benchmark" / "results" / "examples" / "utils" / "current_api"


@lru_cache(maxsize=1)
def load_example_defaults(path: str | Path = DEFAULTS_PATH) -> dict[str, Any]:
    defaults_path = Path(path)
    payload = json.loads(defaults_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Example defaults root must be a mapping: {defaults_path}")
    version = str(payload.get("version", ""))
    if version != DEFAULTS_VERSION:
        raise ValueError(f"Unsupported current API example defaults version: {version}")
    return payload


def build_tsc_suite_config(
    output_dir: str | Path | None = None,
    *,
    persist_on_run: bool = False,
) -> BenchmarkSuiteConfig:
    return _build_suite_config("tsc", TaskType.TS_CLASSIFICATION, output_dir, persist_on_run=persist_on_run)


def build_tser_suite_config(
    output_dir: str | Path | None = None,
    *,
    persist_on_run: bool = False,
) -> BenchmarkSuiteConfig:
    return _build_suite_config("tser", TaskType.TS_REGRESSION, output_dir, persist_on_run=persist_on_run)


def build_forecasting_suite_config(
    output_dir: str | Path | None = None,
    *,
    persist_on_run: bool = False,
) -> BenchmarkSuiteConfig:
    return _build_suite_config("forecasting", TaskType.FORECASTING, output_dir, persist_on_run=persist_on_run)


def run_tsc_example(output_dir: str | Path | None = None, *, persist_on_run: bool = False):
    return run_tsc_benchmark_suite(build_tsc_suite_config(output_dir, persist_on_run=persist_on_run))


def run_tser_example(output_dir: str | Path | None = None, *, persist_on_run: bool = False):
    return run_tser_benchmark_suite(build_tser_suite_config(output_dir, persist_on_run=persist_on_run))


def run_forecasting_example(output_dir: str | Path | None = None, *, persist_on_run: bool = False):
    from benchmark.industrial import run_forecasting_benchmark_suite

    return run_forecasting_benchmark_suite(
        build_forecasting_suite_config(output_dir, persist_on_run=persist_on_run)
    )


def run_all_lightweight_examples(
    output_dir: str | Path | None = None,
    *,
    persist_on_run: bool = False,
) -> dict[str, Any]:
    output_root = Path(output_dir) if output_dir is not None else None
    return {
        "tsc": run_tsc_example(_child_output_dir(output_root, "tsc"), persist_on_run=persist_on_run),
        "tser": run_tser_example(_child_output_dir(output_root, "tser"), persist_on_run=persist_on_run),
    }


def build_kernel_learning_ucr_config_preview(
    output_dir: str | Path | None = None,
    *,
    datasets: tuple[str, ...] = ("Lightning7",),
    persist_on_run: bool = False,
) -> BenchmarkSuiteConfig:
    from benchmark.experiments.kernel_learning.configs import KernelLearningUCRExperimentConfig

    config = KernelLearningUCRExperimentConfig(
        datasets=datasets,
        dataset_limit=len(datasets),
        output_dir=output_dir or DEFAULT_OUTPUT_ROOT / "kernel_learning_ucr_preview",
        persist_on_run=persist_on_run,
        run_name="kernel_learning_ucr_preview",
    )
    return config.build_suite_config()


def _build_suite_config(
    suite_name: str,
    task_type: TaskType,
    output_dir: str | Path | None,
    *,
    persist_on_run: bool,
) -> BenchmarkSuiteConfig:
    defaults = load_example_defaults()
    suites = _mapping(defaults, "suites")
    suite_payload = _mapping(suites, suite_name)
    return BenchmarkSuiteConfig(
        task_type=task_type,
        datasets=(_dataset_spec(defaults, str(suite_payload["dataset"])),),
        models=_model_specs(defaults, str(suite_payload["models"])),
        metrics=tuple(str(metric) for metric in suite_payload["metrics"]),
        artifact_spec=ArtifactSpec(
            output_dir=str(output_dir or DEFAULT_OUTPUT_ROOT / suite_name),
            persist_on_run=persist_on_run,
        ),
        run_spec=RunSpec(**_mapping(suite_payload, "run_spec")),
    )


def _mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload[key]
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping for example defaults key: {key}")
    return value


def _dataset_spec(defaults: dict[str, Any], name: str) -> DatasetSpec:
    return DatasetSpec(**_mapping(_mapping(defaults, "datasets"), name))


def _model_specs(defaults: dict[str, Any], name: str) -> tuple[ModelSpec, ...]:
    payload = _mapping(defaults, "models")[name]
    return tuple(ModelSpec(**item) for item in payload)


def _child_output_dir(output_root: Path | None, child: str) -> Path | None:
    return output_root / child if output_root is not None else None


def _format_result(name: str, result: Any) -> str:
    successful = sum(1 for record in result.run_records if record.status.value == "success")
    primary_metric = result.aggregate_report.primary_metric
    return (
        f"{name}: run_id={result.run_id}, task={result.config.task_type.value}, "
        f"successful_runs={successful}, primary_metric={primary_metric}"
    )


def main() -> int:
    results = run_all_lightweight_examples()
    for name, result in results.items():
        print(_format_result(name, result))
    forecasting_preview = build_forecasting_suite_config()
    print(
        "forecasting_preview: "
        f"datasets={len(forecasting_preview.datasets)}, models={len(forecasting_preview.models)}, "
        f"persist={forecasting_preview.artifact_spec.persist_on_run}"
    )
    preview = build_kernel_learning_ucr_config_preview()
    print(
        "kernel_learning_ucr_preview: "
        f"datasets={len(preview.datasets)}, models={len(preview.models)}, persist={preview.artifact_spec.persist_on_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
