"""Preview contracts plus tiny real-framework smoke runs; no external downloads."""

from dataclasses import replace
import importlib
import json

import numpy as np
import pytest

from examples.release_1_0.config import SCENARIOS, load_config, parse_config


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_config_roundtrip_and_preview_have_no_runtime_effects(scenario, tmp_path, monkeypatch, capsys):
    from examples.release_1_0 import data, models, runner
    from examples.release_1_0.__main__ import main

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Preview must not load data, construct/train a model or write output")

    monkeypatch.setattr(data, "load_holdout", forbidden)
    monkeypatch.setattr(runner, "load_holdout", forbidden)
    monkeypatch.setattr(runner, "run", forbidden)
    monkeypatch.setattr(models, "fit_predict", forbidden)
    importlib.reload(importlib.import_module("examples.release_1_0"))
    config = load_config(scenario)
    assert parse_config(json.loads(json.dumps(config.to_record()))) == config
    output = tmp_path / "must_not_exist"
    assert main(["--scenario", scenario, "--output-dir", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "preview"
    assert not output.exists()


@pytest.mark.parametrize("field,value", [("seed", True), ("seed", -1), ("seed", 10**400), ("backend", "threads"),
                                         ("device", "auto"), ("device", "cuda"), ("extra", 1)])
def test_invalid_common_config_fails_before_effects(field, value):
    raw = load_config().to_record()
    raw[field] = value
    with pytest.raises(ValueError):
        parse_config(raw)


@pytest.mark.parametrize("scenario,key,value", [
    ("motions_graph", "max_iter", False), ("motions_graph", "c", float("nan")),
    ("airline", "horizon", 0), ("airline", "window_size", 100),
    ("motions_neural", "steps", 11), ("motions_neural", "learning_rate", float("inf")),
])
def test_invalid_parameters_are_rejected(scenario, key, value):
    raw = load_config(scenario).to_record()
    raw["parameters"][key] = value
    with pytest.raises(ValueError):
        parse_config(raw)


def test_neural_process_is_not_silently_replaced():
    with pytest.raises(ValueError, match="sequential"):
        load_config("motions_neural", backend="process")


def test_explicit_unavailable_cuda_has_no_cpu_fallback(tiny_holdout, monkeypatch):
    import torch
    from examples.release_1_0.models import fit_predict
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = load_config("motions_neural", device="cuda")
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        fit_predict((config, tiny_holdout.train_x,
                    tiny_holdout.train_y, tiny_holdout.test_x))


def test_normalization_is_train_only_and_does_not_modify_inputs():
    from examples.release_1_0.data import normalize_train_only
    train = np.arange(48, dtype=float).reshape(4, 2, 6)
    snapshot = train.copy()
    a, _, state_a = normalize_train_only(train, train[:2])
    b, changed, state_b = normalize_train_only(train, train[:2] + 10000)
    np.testing.assert_array_equal(train, snapshot)
    np.testing.assert_array_equal(a, b)
    assert state_a == state_b
    np.testing.assert_allclose(a.mean(axis=(0, 2)), 0, atol=1e-7)
    assert changed.mean() > 100


@pytest.fixture
def tiny_holdout():
    # A deterministic test fixture, not the data used by the applied CLI examples.
    from examples.release_1_0.data import Holdout
    rng = np.random.default_rng(10)
    train = rng.normal(size=(8, 2, 24)).astype("float32")
    labels = np.arange(8) % 2
    train[labels == 1] += 2
    test = train[:4].copy() + 0.1
    return Holdout(train, labels, test, labels[:4], ("a", "b", "c", "d"), {"dataset": "test fixture"})


@pytest.mark.parametrize("scenario,backend", [("motions_graph", "sequential"), ("motions_graph", "process"),
                                              ("motions_neural", "sequential"), ("airline", "sequential")])
def test_tiny_actual_fit_predict_and_artifacts(scenario, backend, tiny_holdout, tmp_path):
    from examples.release_1_0.data import Holdout
    from examples.release_1_0.runner import run
    data = tiny_holdout
    config = load_config(scenario, backend=backend)
    if scenario == "motions_neural":
        config = replace(config, parameters=replace(
            config.parameters, steps=1, width=4, batch_size=4))
    if scenario == "airline":
        history = np.sin(np.arange(48) / 4) + np.arange(48) / 20
        config = replace(config, parameters=replace(
            config.parameters, horizon=3, window_size=8))
        data = Holdout(history[:-3], None, history[:-3].copy(), history[-3:],
                       ("45", "46", "47"), {"dataset": "test fixture"})
    snapshot = data.train_x.copy()
    report = run(config, tmp_path, data=data)
    assert report["status"] == "succeeded"
    assert report["test_count"] == len(data.test_y)
    assert all(np.isfinite(value) for value in report["metrics"].values())
    assert json.loads((tmp_path / "result.json").read_text()
                      )["config"] == config.to_record()
    assert len((tmp_path / "predictions.csv").read_text().splitlines()
               ) == len(data.test_y) + 1
    np.testing.assert_array_equal(data.train_x, snapshot)
    if scenario == "motions_neural":
        assert report["training_steps"] == 1
        assert len(report["training_losses"]) == 1
    if scenario == "airline":
        assert (tmp_path / "forecast_pack/plots/multi_model_forecast.png").exists()


def test_neural_replay_restores_rng_and_never_uses_test_labels(tiny_holdout):
    import torch
    from examples.release_1_0.data import normalize_train_only
    from examples.release_1_0.models import fit_predict
    config = load_config("motions_neural")
    config = replace(config, parameters=replace(
        config.parameters, steps=1, width=4))
    train, test, _ = normalize_train_only(
        tiny_holdout.train_x, tiny_holdout.test_x)
    state = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    settings = torch.are_deterministic_algorithms_enabled()
    payload = (config, train, tiny_holdout.train_y, test)
    first, second = fit_predict(payload), fit_predict(payload)
    np.testing.assert_array_equal(first["prediction"], second["prediction"])
    assert first["training_losses"] == second["training_losses"]
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == settings
