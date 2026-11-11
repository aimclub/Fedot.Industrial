"""Relocating legacy policy must not silently change placement semantics."""
import pytest
import torch

from fedot_ind.neural.device import TensorDevicePolicy


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("requested", ["auto", "cuda", "cuda:1", "cpu", "unknown"])
def test_legacy_device_resolution_is_unchanged(monkeypatch, available, requested):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    expected = "cuda:0" if requested == "auto" and available else (
        requested if requested.startswith("cuda") and available else "cpu")
    assert TensorDevicePolicy(requested).resolve_device() == torch.device(expected)


def test_legacy_device_policy_is_one_shared_class():
    from fedot_ind.core.models.ts_forecasting.forecasting_runtime import TensorDevicePolicy as ForecastPolicy
    from fedot_ind.core.models.nn.network_impl.forecasting_model.common import TensorDevicePolicy as NeuralPolicy
    assert TensorDevicePolicy is ForecastPolicy is NeuralPolicy


def test_legacy_dtype_fallback_remains_compatible():
    assert TensorDevicePolicy(dtype="float64").resolve_dtype() is torch.float64
    assert TensorDevicePolicy(dtype="unknown").resolve_dtype() is torch.float32
