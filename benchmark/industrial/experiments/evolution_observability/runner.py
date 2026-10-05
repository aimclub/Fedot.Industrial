"""Reproducible EVO-01 characterization runner."""

from __future__ import annotations

import argparse
import json
import traceback
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from benchmark.industrial.experiments.evolution_observability.contracts import (
    EVOLUTION_BASELINE_SCHEMA,
    EvolutionBaselineConfig,
    EvolutionBaselineScenario,
    EvolutionBaselineTask,
    load_default_evolution_baseline_config,
)


@dataclass(frozen=True)
class EvolutionBaselineResult:
    scenario: str
    task: str
    status: str
    diagnostics_dir: str
    error_type: str | None = None
    error_message: str | None = None

    def to_record(self) -> dict[str, object]:
        record = {
            "scenario": self.scenario,
            "task": self.task,
            "status": self.status,
            "diagnostics_dir": self.diagnostics_dir,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }
        return {key: value for key, value in record.items() if value is not None}


def _read_diagnostics_summary(diagnostics_dir: Path) -> dict[str, object] | None:
    summary_path = diagnostics_dir / "evolution_summary.json"
    if not summary_path.is_file():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def build_evolution_baseline_report(
        config: EvolutionBaselineConfig,
        results: tuple[EvolutionBaselineResult, ...],
        output_dir: str | Path,
        *,
        execution_mode: str,
) -> dict[str, object]:
    """Join suite outcomes with available diagnostics into one report."""
    root = Path(output_dir)
    scenario_by_name = {
        scenario.name: scenario for scenario in config.scenarios}
    report_rows = []
    for result in results:
        scenario = scenario_by_name[result.scenario]
        diagnostics_dir = Path(result.diagnostics_dir)
        summary = _read_diagnostics_summary(diagnostics_dir)
        artifacts = {
            name: str(path)
            for name, path in {
                "events_jsonl": diagnostics_dir / "evolution_events.jsonl",
                "summary_json": diagnostics_dir / "evolution_summary.json",
                "summary_markdown": diagnostics_dir / "evolution_summary.md",
            }.items()
            if path.is_file()
        }
        report_rows.append({
            **result.to_record(),
            "scenario_contract": scenario.to_record(),
            "diagnostics": summary,
            "artifacts": artifacts,
        })

    status_counts = Counter(result.status for result in results)
    return {
        "schema_version": EVOLUTION_BASELINE_SCHEMA,
        "execution_mode": execution_mode,
        "suite_contract": config.to_record(),
        "status_counts": dict(sorted(status_counts.items())),
        "scenarios": report_rows,
        "artifacts": {
            "manifest": str(root / "baseline_manifest.json"),
            "resolved_suite": str(root / "resolved_suite.json"),
        },
    }


def _baseline_report_markdown(report: Mapping[str, object]) -> str:
    rows = []
    for scenario in report["scenarios"]:
        diagnostics = scenario.get("diagnostics") or {}
        failure = scenario.get("error_type") or "-"
        rows.append(
            "| {scenario} | {task} | {status} | {attempts} | {accepted} | "
            "{failed} | {failure} |".format(
                scenario=scenario["scenario"],
                task=scenario["task"],
                status=scenario["status"],
                attempts=diagnostics.get("mutation_attempts", "-"),
                accepted=diagnostics.get("candidates_accepted", "-"),
                failed=diagnostics.get("evaluations_failed", "-"),
                failure=failure,
            )
        )
    return "\n".join((
        "# Evolution baseline report",
        "",
        f"Schema: `{report['schema_version']}`",
        f"Execution mode: `{report['execution_mode']}`",
        "",
        "| Scenario | Task | Status | Mutation attempts | Accepted candidates | "
        "Failed evaluations | Failure type |",
        "|---|---|---|---:|---:|---:|---|",
        *rows,
        "",
        "The JSON companion contains the resolved suite contract, diagnostic summaries, "
        "failure messages and source artifact paths.",
        "",
    ))


def write_evolution_baseline_report(
        report: Mapping[str, object],
        output_dir: str | Path,
) -> tuple[Path, Path]:
    """Persist the aggregate baseline report in machine and review formats."""
    root = Path(output_dir)
    json_path = root / "baseline_summary.json"
    markdown_path = root / "baseline_summary.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        _baseline_report_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def build_synthetic_input(
        scenario: EvolutionBaselineScenario,
        *,
        seed: int,
):
    """Build deterministic data with a learnable signal for one task."""
    rng = np.random.default_rng(seed)
    time_axis = np.linspace(
        0.0, 2.0 * np.pi, scenario.series_length, endpoint=False)
    if scenario.task is EvolutionBaselineTask.FORECASTING:
        trend = np.linspace(0.0, 1.5, scenario.series_length)
        seasonal = np.sin(time_axis) + 0.25 * np.sin(3.0 * time_axis)
        return trend + seasonal + rng.normal(0.0, 0.03, scenario.series_length)

    amplitudes = rng.uniform(0.7, 1.3, size=scenario.sample_count)
    phases = rng.uniform(-0.4, 0.4, size=scenario.sample_count)
    features = np.stack([
        amplitude * np.sin(time_axis + phase)
        + 0.15 * np.cos(2.0 * time_axis - phase)
        + rng.normal(0.0, 0.04, scenario.series_length)
        for amplitude, phase in zip(amplitudes, phases)
    ])[:, np.newaxis, :]
    signal = features[:, 0, : scenario.series_length // 2].mean(axis=1)
    if scenario.task is EvolutionBaselineTask.CLASSIFICATION:
        target = (signal > np.median(signal)).astype(int)
    else:
        target = 2.5 * signal + 0.3 * amplitudes
    return features, target


def build_fedot_industrial_config(
        config: EvolutionBaselineConfig,
        scenario: EvolutionBaselineScenario,
        output_dir: Path,
) -> dict[str, object]:
    """Build the public API configuration used by the characterization run."""
    from fedot_ind.core.repository.config_repository import (
        DEFAULT_CLF_API_CONFIG,
        DEFAULT_REG_API_CONFIG,
        DEFAULT_TSF_API_CONFIG,
    )

    template_by_task = {
        EvolutionBaselineTask.CLASSIFICATION: DEFAULT_CLF_API_CONFIG,
        EvolutionBaselineTask.REGRESSION: DEFAULT_REG_API_CONFIG,
        EvolutionBaselineTask.FORECASTING: DEFAULT_TSF_API_CONFIG,
    }
    api_config = deepcopy(template_by_task[scenario.task])
    diagnostics_dir = output_dir / scenario.name / "diagnostics"
    task_output = output_dir / scenario.name / "runtime"
    api_config["automl_config"]["optimisation_strategy"] = {
        "optimisation_agent": "Industrial",
        "optimisation_strategy": {
            "mutation_agent": "random",
            "mutation_strategy": "growth_mutation_strategy",
            "diagnostics_output_dir": str(diagnostics_dir),
        },
    }
    if scenario.task is EvolutionBaselineTask.FORECASTING:
        horizon = int(scenario.forecast_horizon)
        api_config["automl_config"]["task_params"] = {
            "forecast_length": horizon}
        api_config["industrial_config"]["task_params"] = {
            "forecast_length": horizon}
    api_config["learning_config"]["learning_strategy_params"] = {
        **api_config["learning_config"].get("learning_strategy_params", {}),
        "timeout": config.timeout_minutes,
        "pop_size": config.population_size,
        "num_of_generations": config.generations,
        "n_jobs": 1,
        "with_tuning": False,
        "seed": config.seed,
    }
    api_config["learning_config"]["optimisation_loss"] = {
        "quality_loss": scenario.quality_metric,
    }
    api_config["compute_config"] = {
        **api_config["compute_config"],
        "backend": "cpu",
        "distributed": {
            "processes": False,
            "n_workers": 1,
            "threads_per_worker": 1,
            "memory_limit": 0.3,
        },
        "output_folder": str(task_output),
        "automl_folder": {
            "optimisation_history": str(task_output / "opt_hist"),
            "composition_results": str(task_output / "comp_res"),
        },
    }
    return api_config


def _default_model_factory(api_config: Mapping[str, object]):
    from fedot_ind.api.main import FedotIndustrial

    return FedotIndustrial(**dict(api_config))


def execute_evolution_baseline_scenario(
        config: EvolutionBaselineConfig,
        scenario: EvolutionBaselineScenario,
        output_dir: str | Path,
        *,
        model_factory: Callable[[Mapping[str, object]],
                                object] = _default_model_factory,
) -> EvolutionBaselineResult:
    """Execute one scenario and retain failures as baseline evidence."""
    root = Path(output_dir)
    diagnostics_dir = root / scenario.name / "diagnostics"
    api_config = build_fedot_industrial_config(config, scenario, root)
    input_data = build_synthetic_input(scenario, seed=config.seed)
    model = None
    try:
        model = model_factory(api_config)
        model.fit(input_data)
        summary_path = diagnostics_dir / "evolution_summary.json"
        if not summary_path.is_file():
            return EvolutionBaselineResult(
                scenario=scenario.name,
                task=scenario.task.value,
                status="not_observed",
                diagnostics_dir=str(diagnostics_dir),
                error_type="MissingDiagnosticsArtifact",
                error_message=(
                    "FEDOT completed fitting without invoking the Industrial evolutionary optimizer"
                ),
            )
        return EvolutionBaselineResult(
            scenario=scenario.name,
            task=scenario.task.value,
            status="succeeded",
            diagnostics_dir=str(diagnostics_dir),
        )
    except Exception as error:
        failure_dir = root / scenario.name
        failure_dir.mkdir(parents=True, exist_ok=True)
        (failure_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return EvolutionBaselineResult(
            scenario=scenario.name,
            task=scenario.task.value,
            status="failed",
            diagnostics_dir=str(diagnostics_dir),
            error_type=type(error).__name__,
            error_message=str(error),
        )
    finally:
        if model is not None and hasattr(model, "shutdown"):
            model.shutdown()


def run_evolution_baseline_suite(
        config: EvolutionBaselineConfig,
        output_dir: str | Path,
        *,
        execute: bool = False,
        model_factory: Callable[[Mapping[str, object]],
                                object] = _default_model_factory,
) -> tuple[EvolutionBaselineResult, ...]:
    """Resolve or execute all fixed EVO-01 scenarios."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "resolved_suite.json").write_text(
        json.dumps(config.to_record(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if execute:
        results = tuple(
            execute_evolution_baseline_scenario(
                config,
                scenario,
                root,
                model_factory=model_factory,
            )
            for scenario in config.scenarios
        )
    else:
        results = tuple(
            EvolutionBaselineResult(
                scenario=scenario.name,
                task=scenario.task.value,
                status="planned",
                diagnostics_dir=str(root / scenario.name / "diagnostics"),
            )
            for scenario in config.scenarios
        )
    manifest = {
        "schema_version": EVOLUTION_BASELINE_SCHEMA,
        "execution_mode": "execute" if execute else "dry_run",
        "results": [result.to_record() for result in results],
    }
    (root / "baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = build_evolution_baseline_report(
        config,
        results,
        root,
        execution_mode=manifest["execution_mode"],
    )
    write_evolution_baseline_report(report, root)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the EVO-01 characterization suite")
    parser.add_argument("--output-dir", default="results/evolution_baseline")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    results = run_evolution_baseline_suite(
        load_default_evolution_baseline_config(),
        args.output_dir,
        execute=args.execute,
    )
    print(json.dumps([result.to_record() for result in results], indent=2))
    return 0 if all(result.status != "failed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
