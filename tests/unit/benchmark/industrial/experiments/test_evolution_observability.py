import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from benchmark.industrial.experiments.evolution_observability import (
    EvolutionBaselineConfig,
    EvolutionBaselineScenario,
    EvolutionBaselineTask,
    build_fedot_industrial_config,
    build_synthetic_input,
    load_default_evolution_baseline_config,
    run_evolution_baseline_suite,
)
from fedot_ind.core.repository.config_repository import DEFAULT_CLF_API_CONFIG


def test_default_evolution_suite_covers_three_primary_tasks():
    config = load_default_evolution_baseline_config()

    assert [scenario.task for scenario in config.scenarios] == [
        EvolutionBaselineTask.CLASSIFICATION,
        EvolutionBaselineTask.REGRESSION,
        EvolutionBaselineTask.FORECASTING,
    ]
    assert config.population_size >= 2
    assert config.generations == 1


def test_synthetic_inputs_are_deterministic_and_task_shaped():
    config = load_default_evolution_baseline_config()

    classification, regression, forecasting = (
        build_synthetic_input(scenario, seed=config.seed)
        for scenario in config.scenarios
    )
    repeated_classification = build_synthetic_input(
        config.scenarios[0], seed=config.seed)

    np.testing.assert_allclose(classification[0], repeated_classification[0])
    np.testing.assert_array_equal(
        classification[1], repeated_classification[1])
    assert classification[0].shape == (24, 1, 32)
    assert set(np.unique(classification[1])) == {0, 1}
    assert regression[0].shape == (24, 1, 32)
    assert regression[1].shape == (24,)
    assert forecasting.shape == (96,)


def test_public_api_config_enables_diagnostics_without_mutating_defaults(tmp_path):
    defaults_before = deepcopy(DEFAULT_CLF_API_CONFIG)
    config = load_default_evolution_baseline_config()
    scenario = config.scenarios[0]

    api_config = build_fedot_industrial_config(config, scenario, tmp_path)

    strategy = api_config["automl_config"]["optimisation_strategy"]["optimisation_strategy"]
    assert Path(strategy["diagnostics_output_dir"]).parts[-2:] == (
        "tiny_classification", "diagnostics")
    assert api_config["learning_config"]["learning_strategy_params"]["num_of_generations"] == 1
    assert api_config["compute_config"]["distributed"]["n_workers"] == 1
    assert DEFAULT_CLF_API_CONFIG == defaults_before


def test_dry_run_writes_resolved_contract_without_constructing_models(tmp_path):
    config = load_default_evolution_baseline_config()

    def unexpected_factory(api_config):
        raise AssertionError(f"dry run constructed a model: {api_config}")

    results = run_evolution_baseline_suite(
        config,
        tmp_path,
        execute=False,
        model_factory=unexpected_factory,
    )

    manifest = json.loads(
        (tmp_path / "baseline_manifest.json").read_text(encoding="utf-8"))
    resolved = json.loads(
        (tmp_path / "resolved_suite.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (tmp_path / "baseline_summary.json").read_text(encoding="utf-8"))
    assert [result.status for result in results] == [
        "planned", "planned", "planned"]
    assert manifest["execution_mode"] == "dry_run"
    assert resolved == config.to_record()
    assert summary["status_counts"] == {"planned": 3}
    assert (tmp_path / "baseline_summary.md").is_file()


def test_execute_mode_runs_each_scenario_and_closes_models(tmp_path):
    config = load_default_evolution_baseline_config()
    models = []

    class FakeModel:
        def __init__(self, api_config):
            self.api_config = api_config
            self.fitted = False
            self.closed = False
            models.append(self)

        def fit(self, input_data):
            self.fitted = True
            assert input_data is not None
            diagnostics = Path(
                self.api_config["automl_config"]["optimisation_strategy"]
                ["optimisation_strategy"]["diagnostics_output_dir"]
            )
            diagnostics.mkdir(parents=True, exist_ok=True)
            (diagnostics / "evolution_summary.json").write_text("{}", encoding="utf-8")

        def shutdown(self):
            self.closed = True

    results = run_evolution_baseline_suite(
        config,
        tmp_path,
        execute=True,
        model_factory=FakeModel,
    )

    assert [result.status for result in results] == ["succeeded"] * 3
    assert len(models) == 3
    assert all(model.fitted and model.closed for model in models)
    summary = json.loads(
        (tmp_path / "baseline_summary.json").read_text(encoding="utf-8"))
    assert summary["status_counts"] == {"succeeded": 3}
    assert all(row["diagnostics"] == {} for row in summary["scenarios"])
    assert all("summary_json" in row["artifacts"]
               for row in summary["scenarios"])


def test_execute_mode_preserves_failure_evidence_and_continues_suite(tmp_path):
    config = load_default_evolution_baseline_config()
    calls = 0

    class FailingFirstModel:
        def __init__(self, api_config):
            nonlocal calls
            calls += 1
            self.should_fail = calls == 1
            self.closed = False
            self.api_config = api_config

        def fit(self, input_data):
            if self.should_fail:
                raise RuntimeError("characterized failure")
            diagnostics = Path(
                self.api_config["automl_config"]["optimisation_strategy"]
                ["optimisation_strategy"]["diagnostics_output_dir"]
            )
            diagnostics.mkdir(parents=True, exist_ok=True)
            (diagnostics / "evolution_summary.json").write_text("{}", encoding="utf-8")

        def shutdown(self):
            self.closed = True

    results = run_evolution_baseline_suite(
        config,
        tmp_path,
        execute=True,
        model_factory=FailingFirstModel,
    )

    assert [result.status for result in results] == [
        "failed", "succeeded", "succeeded"]
    assert results[0].error_type == "RuntimeError"
    assert "characterized failure" in (tmp_path / "tiny_classification" / "failure.txt").read_text(
        encoding="utf-8"
    )
    summary = json.loads(
        (tmp_path / "baseline_summary.json").read_text(encoding="utf-8"))
    failed = summary["scenarios"][0]
    assert failed["error_type"] == "RuntimeError"
    assert failed["diagnostics"] is None


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"schema_version": "other"}, "Unsupported"),
        ({
            "schema_version": "industrial_evolution_baseline@1",
            "seed": 1,
            "timeout_minutes": 1,
            "population_size": 2,
            "generations": 1,
            "scenarios": [],
        }, "At least one"),
    ],
)
def test_invalid_suite_configuration_fails_at_boundary(payload, message):
    with pytest.raises((KeyError, ValueError), match=message):
        EvolutionBaselineConfig.from_mapping(payload)


def test_non_forecasting_scenario_rejects_forecast_horizon():
    scenario = EvolutionBaselineScenario(
        name="invalid",
        task=EvolutionBaselineTask.CLASSIFICATION,
        sample_count=2,
        series_length=8,
        forecast_horizon=2,
        quality_metric="f1",
    )

    with pytest.raises(ValueError, match="only valid for forecasting"):
        scenario.validate()
