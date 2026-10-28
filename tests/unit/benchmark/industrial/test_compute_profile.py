"""Profiling is reproducible, explicit and does not change process Torch flags."""
import copy
import json
from pathlib import Path

import pytest
import torch

from benchmark.industrial.experiments.compute_profile.__main__ import load_settings, run


def _settings():
    return {"seed": 1729, "warmup": 0, "repeats": 1, "rtol": 1e-5, "atol": 1e-5,
            "memory_budget_bytes": 1048576,
            "cases": [{"operation": "digitize", "batch": 2, "length": 9, "bins": 4},
                      {"operation": "mtf", "batch": 2, "length": 17, "bins": 4, "image_size": 5}]}


def test_defaults_are_valid():
    import benchmark.industrial.experiments.compute_profile as package
    assert load_settings(Path(package.__file__).with_name("defaults.json"))["repeats"] >= 1


@pytest.mark.parametrize("key,value", [("seed", True), ("warmup", -1), ("repeats", 0),
                                       ("atol", float("nan")), ("rtol", True), ("extra", 1)])
def test_invalid_profile_settings_fail_before_execution(tmp_path, key, value):
    raw = _settings()
    raw[key] = value
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_settings(path)


def test_cpu_profile_counts_transfers_and_preserves_torch_state():
    raw = _settings()
    before = copy.deepcopy(raw)
    rng = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads(), torch.get_num_interop_threads()
    precision = torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32
    rows = run(raw, ["cpu"])
    assert len(rows) == 8
    assert all(row["within_tolerance"] and row["transfer_bytes"] == 0 for row in rows)
    assert all(row["median_seconds"] > 0 and row["input_bytes"] > 0 for row in rows)
    assert raw == before
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert (torch.get_num_threads(), torch.get_num_interop_threads()) == threads
    assert (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32) == precision


def test_explicit_cuda_never_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        run(_settings(), ["cuda"])


def test_public_mtf_execution_uses_admitted_plan():
    from benchmark.industrial.experiments.compute_profile.placement import probe_mtf_placement
    result = probe_mtf_placement(_settings(), "cpu")
    assert result["within_tolerance"]
    assert result["memory_estimate"]["transfer_bytes"] == 0
    assert result["backend"] == "sequential"


@pytest.mark.parametrize("raw", [[], None, True, {**_settings(), "cases": [None]},
                                 {**_settings(), "cases": [{"operation": "mtf", "batch": 1,
                                                           "length": 2, "bins": 1, "image_size": 1}]}])
def test_malformed_root_and_cases_are_rejected(tmp_path, raw):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_settings(path)


def test_mtf_transition_matrix_budget_is_rejected_before_execution(monkeypatch):
    from benchmark.industrial.experiments.compute_profile import placement
    raw = _settings()
    raw["memory_budget_bytes"] = 32768
    raw["cases"] = [{"operation": "mtf", "batch": 1, "length": 2, "bins": 256, "image_size": 2}]

    def forbidden(*args, **kwargs):
        pytest.fail("Insufficient memory must be rejected before MTF execution")

    monkeypatch.setattr(placement.MTF, "transform", forbidden)
    with pytest.raises(ValueError, match="memory_budget_exceeded"):
        placement.probe_mtf_placement(raw, "cpu")
