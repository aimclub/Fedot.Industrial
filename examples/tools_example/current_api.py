from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.experiments.kernel_learning.configs import KernelLearningUCRExperimentConfig
from benchmark.industrial import BenchmarkSuiteConfig, run_tsc_benchmark_suite, run_tser_benchmark_suite
from examples.tools_example.registry import invoke_tool, list_tool_specs
from examples.tools_example.services.anomaly import anomaly_detection_context
from examples.tools_example.services.common import config_summary, load_tool_defaults, result_summary
from examples.tools_example.services.training import build_training_config


def run_classification_baseline(output_dir: str | Path | None = None):
    config = build_training_config(
        {"task_type": "ts_classification", "dataset_name": "Lightning7", "output_dir": output_dir},
        persist_on_run=False,
    )
    return run_tsc_benchmark_suite(config)


def run_regression_baseline(output_dir: str | Path | None = None):
    config = build_training_config(
        {"task_type": "ts_regression", "dataset_name": "NaturalGasPricesSentiment", "output_dir": output_dir},
        persist_on_run=False,
    )
    return run_tser_benchmark_suite(config)


def build_forecasting_preview(output_dir: str | Path | None = None) -> BenchmarkSuiteConfig:
    return build_training_config(
        {"task_type": "forecasting", "subset": "daily", "sample_size": 3, "output_dir": output_dir},
        persist_on_run=False,
    )


def build_kernel_learning_preview(output_dir: str | Path | None = None) -> BenchmarkSuiteConfig:
    defaults = load_tool_defaults()
    dataset_name = str(defaults["kernel_learning"]["dataset_name"])
    config = KernelLearningUCRExperimentConfig(
        datasets=(dataset_name,),
        output_dir=output_dir,
        persist_on_run=False,
        run_name="kernel_learning_ucr_preview",
    )
    return config.build_suite_config()


def local_data_context(name: str) -> dict[str, Any]:
    defaults = load_tool_defaults()
    payload = dict(defaults["local_data"][name])
    root = Path(payload["data_root"])
    project_root = Path(__file__).resolve().parents[2]
    payload["data_root"] = str(root if root.is_absolute() else project_root / root)
    return payload


def main() -> int:
    import json

    print(json.dumps({"tools": list_tool_specs()}, indent=2))
    return 0


__all__ = [
    "anomaly_detection_context",
    "build_forecasting_preview",
    "build_kernel_learning_preview",
    "config_summary",
    "invoke_tool",
    "list_tool_specs",
    "load_tool_defaults",
    "local_data_context",
    "result_summary",
    "run_classification_baseline",
    "run_regression_baseline",
]


if __name__ == "__main__":
    raise SystemExit(main())
