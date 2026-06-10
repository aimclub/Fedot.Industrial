from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark.industrial import (
    ArtifactRecord,
    ModelSpec,
    ResultAnalysisSpec,
    render_benchmark_result_analysis_pack,
    render_forecast_comparison_pack,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[2]
DEFAULTS_PATH = PACKAGE_ROOT / "scenario_defaults.json"
DEFAULTS_VERSION = "industrial_domain_scenarios@1"


@lru_cache(maxsize=1)
def load_scenario_defaults(path: str | Path = DEFAULTS_PATH) -> dict[str, Any]:
    defaults_path = Path(path)
    payload = json.loads(defaults_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Scenario defaults root must be a mapping: {defaults_path}")
    version = str(payload.get("version", ""))
    if version != DEFAULTS_VERSION:
        raise ValueError(f"Unsupported scenario defaults version: {version}")
    return payload


def list_domain_scenarios(domain: str | None = None) -> tuple[str, ...]:
    scenarios = load_scenario_defaults()["scenarios"]
    names = sorted(scenarios)
    if domain is not None:
        names = [name for name in names if scenarios[name]["domain"] == domain]
    return tuple(names)


def build_scenario_model_specs(scenario_name: str) -> tuple[ModelSpec, ...]:
    scenario = _scenario_payload(scenario_name)
    local_specs = tuple(ModelSpec(**payload) for payload in scenario["models"])
    reference_specs = build_kernel_learning_reference_model_specs(str(scenario["task_type"]))
    return _dedupe_model_specs((*reference_specs, *local_specs))


def build_kernel_learning_reference_model_specs(task_type: str) -> tuple[ModelSpec, ...]:
    from benchmark.experiments.kernel_learning.configs import (
        build_forecasting_kernel_learning_models,
        build_tser_kernel_learning_models,
        build_ucr_kernel_learning_models,
    )

    if task_type == "ts_classification":
        return build_ucr_kernel_learning_models()
    if task_type == "ts_regression":
        return build_tser_kernel_learning_models()
    if task_type == "forecasting":
        return build_forecasting_kernel_learning_models()
    return ()


def build_scenario_context(scenario_name: str) -> dict[str, Any]:
    scenario = _scenario_payload(scenario_name)
    data_path = PROJECT_ROOT / scenario["data_path"]
    artifact_root = PROJECT_ROOT / load_scenario_defaults()["artifact_root"] / scenario_name
    models = build_scenario_model_specs(scenario_name)
    return {
        "scenario": scenario_name,
        "domain": scenario["domain"],
        "task_type": scenario["task_type"],
        "entrypoint": str(PACKAGE_ROOT / scenario["entrypoint"]),
        "data_path": str(data_path),
        "data_exists_locally": data_path.exists(),
        "artifact_root": str(artifact_root),
        "metric_name": scenario["metric_name"],
        "metric_direction": scenario["metric_direction"],
        "feature_generators": tuple(load_scenario_defaults()["feature_generators"]),
        "models": tuple(model.display_name for model in models),
        "model_specs": models,
        "visualization_contract": "benchmark.industrial visualization/evaluation artifact packs",
    }


def build_scenario_preview_frame(scenario_name: str) -> pd.DataFrame:
    scenario = _scenario_payload(scenario_name)
    models = build_scenario_model_specs(scenario_name)
    lower_is_better = scenario["metric_direction"] == "lower"
    rows: list[dict[str, Any]] = []
    for dataset_index in range(1, 3):
        row: dict[str, Any] = {"dataset_name": f"{scenario_name}_preview_{dataset_index}"}
        for model_index, model in enumerate(models):
            if lower_is_better:
                row[model.display_name] = round(0.35 + 0.04 * model_index + 0.03 * dataset_index, 4)
            else:
                row[model.display_name] = round(0.92 - 0.04 * model_index - 0.02 * dataset_index, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def render_scenario_preview_pack(
        scenario_name: str,
        output_dir: str | Path | None = None,
) -> tuple[ArtifactRecord, ...]:
    scenario = _scenario_payload(scenario_name)
    target_dir = Path(output_dir) if output_dir is not None else (
        PROJECT_ROOT / load_scenario_defaults()["artifact_root"] / scenario_name
    )
    models = build_scenario_model_specs(scenario_name)
    target_model = models[0].display_name if models else None
    spec = ResultAnalysisSpec(
        metric_name=str(scenario["metric_name"]),
        metric_direction=str(scenario["metric_direction"]),
        source_label=f"{scenario_name}_current_api_preview",
        task_type=str(scenario["task_type"]),
    )
    return render_benchmark_result_analysis_pack(
        build_scenario_preview_frame(scenario_name),
        target_dir,
        spec=spec,
        target_model=target_model,
    )


def build_scenario_forecast_preview(scenario_name: str) -> dict[str, Any]:
    scenario = _scenario_payload(scenario_name)
    if scenario["task_type"] != "forecasting":
        raise ValueError(f"Scenario {scenario_name!r} is not a forecasting scenario.")

    history_length = 36
    horizon = 12
    steps = list(range(history_length + horizon))
    signal = [
        10.0 + 0.14 * step + 1.8 * _sin_like(step / 3.5) + 0.7 * _sin_like(step / 1.9)
        for step in steps
    ]
    history = tuple(round(value, 4) for value in signal[:history_length])
    actual = tuple(round(value, 4) for value in signal[history_length:])
    forecasts: dict[str, tuple[float, ...]] = {}
    for model_index, model in enumerate(build_scenario_model_specs(scenario_name)):
        strength = max(0.05, 0.24 - 0.035 * model_index)
        bias = 0.28 - 0.08 * model_index
        model_forecast = [
            actual_value + bias + strength * _sin_like((step + model_index) / 2.2)
            for step, actual_value in enumerate(actual)
        ]
        forecasts[model.display_name] = tuple(round(value, 4) for value in model_forecast)
    return {
        "scenario": scenario_name,
        "series_id": f"{scenario_name}_forecast_preview",
        "history": history,
        "actual": actual,
        "forecasts": forecasts,
    }


def render_scenario_forecast_pack(
        scenario_name: str,
        output_dir: str | Path | None = None,
) -> tuple[ArtifactRecord, ...]:
    preview = build_scenario_forecast_preview(scenario_name)
    target_dir = Path(output_dir) if output_dir is not None else (
        PROJECT_ROOT / load_scenario_defaults()["artifact_root"] / scenario_name / "forecast_comparison"
    )
    return render_forecast_comparison_pack(
        history=preview["history"],
        actual=preview["actual"],
        forecasts=preview["forecasts"],
        output_dir=target_dir,
        title=f"{scenario_name}: multi-model forecast preview",
        series_id=str(preview["series_id"]),
    )


def preflight_summary() -> dict[str, Any]:
    scenarios = list_domain_scenarios()
    return {
        "version": load_scenario_defaults()["version"],
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "feature_generators": tuple(load_scenario_defaults()["feature_generators"]),
    }


def _scenario_payload(scenario_name: str) -> dict[str, Any]:
    scenarios = load_scenario_defaults()["scenarios"]
    if scenario_name not in scenarios:
        raise ValueError(f"Unknown industrial example scenario: {scenario_name}")
    return dict(scenarios[scenario_name])


def _dedupe_model_specs(model_specs: tuple[ModelSpec, ...]) -> tuple[ModelSpec, ...]:
    deduped: list[ModelSpec] = []
    seen: set[str] = set()
    for model in model_specs:
        if model.display_name in seen:
            continue
        deduped.append(model)
        seen.add(model.display_name)
    return tuple(deduped)


def _sin_like(value: float) -> float:
    import math

    return float(math.sin(value))


if __name__ == "__main__":
    print(preflight_summary())


__all__ = [
    "build_scenario_context",
    "build_kernel_learning_reference_model_specs",
    "build_scenario_forecast_preview",
    "build_scenario_model_specs",
    "build_scenario_preview_frame",
    "list_domain_scenarios",
    "load_scenario_defaults",
    "preflight_summary",
    "render_scenario_forecast_pack",
    "render_scenario_preview_pack",
]
