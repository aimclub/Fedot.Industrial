import json
from copy import deepcopy
from unittest.mock import Mock

import pytest

from benchmark.industrial.experiments.evolution_observability.contracts import (
    load_default_evolution_baseline_config,
)
from benchmark.industrial.experiments.evolution_observability.runner import (
    build_fedot_industrial_config,
)
from benchmark.industrial.experiments.mutation_planning.api_runner import (
    build_api_verification_config,
    load_api_verification_policy,
    main,
    run_api_verification,
)
from fedot_ind.core.optimizer.configuration import normalize_evolution_config


@pytest.mark.parametrize("disable_cache", [False, True])
def test_configuration_preserves_input_and_resolves_policy(tmp_path, disable_cache):
    suite = load_default_evolution_baseline_config()
    base = build_fedot_industrial_config(suite, suite.scenarios[0], tmp_path)
    before = deepcopy(base)
    resolved = build_api_verification_config(
        base, load_api_verification_policy(), without_prediction_cache=disable_cache,
    )
    assert base == before
    strategy = resolved["automl_config"]["optimisation_strategy"]["optimisation_strategy"]
    policy = normalize_evolution_config(strategy)
    assert policy.random_seed == suite.seed
    assert policy.mutation_planning.enabled
    assert policy.execution_policy.diagnostics_output_dir == str(
        tmp_path / suite.scenarios[0].name / "diagnostics"
    )
    params = resolved["learning_config"]["learning_strategy_params"]
    if disable_cache:
        assert params["use_predictions_cache"] is False
    else:
        assert params == before["learning_config"]["learning_strategy_params"]


@pytest.mark.parametrize("legacy", [False, True])
def test_preview_never_constructs_models_and_persists_exact_config(tmp_path, legacy):
    factory = Mock(side_effect=AssertionError("preview must not train"))
    results = run_api_verification(tmp_path, legacy=legacy, model_factory=factory)
    factory.assert_not_called()
    assert all(result.status == "planned" for result in results)
    settings = json.loads((tmp_path / "verification_settings.json").read_text())
    assert settings["execute"] is False
    assert settings["evolution_policy"]["mutation_planning"]["enabled"] is not legacy
    for result in results:
        record = json.loads((tmp_path / f"{result.scenario}_api_config.json").read_text())
        strategy = record["automl_config"]["optimisation_strategy"]["optimisation_strategy"]
        assert strategy["mutation_planning"]["enabled"] is not legacy
    assert not list(tmp_path.rglob("runtime"))


def test_execution_reports_real_failure_and_shuts_down(tmp_path):
    models = []

    def factory(config):
        strategy = config["automl_config"]["optimisation_strategy"]["optimisation_strategy"]
        assert strategy["mutation_planning"]["enabled"]
        assert config["learning_config"]["learning_strategy_params"]["use_predictions_cache"] is False
        model = Mock()
        model.fit.side_effect = ValueError("native input split failure")
        models.append(model)
        return model

    results = run_api_verification(
        tmp_path, execute=True, without_prediction_cache=True, model_factory=factory,
    )
    assert len(models) == len(results) == 3
    for model, result in zip(models, results):
        model.shutdown.assert_called_once()
        assert result.status == "failed"
        assert result.error_type == "ValueError"
        assert result.error_message == "native input split failure"
    report = json.loads((tmp_path / "baseline_summary.json").read_text())
    assert report["status_counts"] == {"failed": 3}


def test_cli_preview_succeeds(tmp_path):
    assert main(["--output-dir", str(tmp_path)]) == 0


def test_execution_does_not_accept_previous_diagnostics(tmp_path):
    suite = load_default_evolution_baseline_config()
    marker = tmp_path / suite.scenarios[0].name / "diagnostics" / "evolution_summary.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('{"previous_run": true}')
    factory = Mock()
    with pytest.raises(ValueError, match="fresh output directory"):
        run_api_verification(tmp_path, execute=True, model_factory=factory)
    factory.assert_not_called()
    assert marker.read_text() == '{"previous_run": true}'


def test_cli_missing_diagnostics_is_not_success(tmp_path, monkeypatch):
    from benchmark.industrial.experiments.evolution_observability.runner import EvolutionBaselineResult

    monkeypatch.setattr(
        "benchmark.industrial.experiments.mutation_planning.api_runner.run_api_verification",
        lambda *args, **kwargs: (EvolutionBaselineResult("tiny", "classification", "not_observed", "none"),),
    )
    assert main(["--output-dir", str(tmp_path), "--execute"]) == 1


def test_preview_records_explicit_compatibility_audit(tmp_path, monkeypatch):
    audit = {"worktree": str(tmp_path / "fedot"), "patches": [{"name": "verified-fix"}]}
    monkeypatch.setattr("fedot_ind.integration.fedot.compat.audit_compatibility_worktree", lambda _: audit)
    run_api_verification(tmp_path, compatibility_worktree=tmp_path / "fedot")
    settings = json.loads((tmp_path / "verification_settings.json").read_text())
    assert settings["fedot_compatibility"] == audit


def test_execution_rejects_active_fedot_outside_audited_worktree(tmp_path, monkeypatch):
    audit = {"worktree": str(tmp_path / "other"), "patches": []}
    monkeypatch.setattr("fedot_ind.integration.fedot.compat.audit_compatibility_worktree", lambda _: audit)
    factory = Mock()
    with pytest.raises(ValueError, match="Active FEDOT import"):
        run_api_verification(tmp_path, execute=True, compatibility_worktree=tmp_path, model_factory=factory)
    factory.assert_not_called()
    assert not (tmp_path / "verification_settings.json").exists()
