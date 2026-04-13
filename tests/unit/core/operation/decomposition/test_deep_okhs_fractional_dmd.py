from types import SimpleNamespace

import numpy as np
import torch

from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.fractional_dmd import (
    FractionalDMD,
)
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.policies import (
    StabilityPolicy,
)


def _build_fdmd(device="cpu"):
    liouville = SimpleNamespace(
        okhs=SimpleNamespace(q=0.8, dt=0.5),
        eigenvalues_=np.array([-0.05 + 0.02j, -0.05 - 0.02j], dtype=np.complex128),
    )
    fdmd = FractionalDMD(
        liouville_operator=liouville,
        stability_policy=StabilityPolicy(),
        device=device,
    )
    fdmd.modes_ = np.array([[1.0], [1.0]], dtype=float)
    return fdmd


def test_prepare_prediction_state_coerces_mixed_inputs_to_cpu_tensors():
    fdmd = _build_fdmd()

    state = fdmd._prepare_prediction_state(
        initial_trajectory=[1.0, 0.9, 0.8, 0.75],
        t_span=[2.0, 2.5, 3.0],
        stability_threshold=0.01,
        prediction_mode_selection_policy="adaptive_tail_energy",
        max_prediction_modes=2,
        min_prediction_modes=1,
    )

    tensor_keys = (
        "initial_trajectory",
        "t_span",
        "eig_full",
        "stable_mask",
        "eig",
        "xi",
        "selected_mode_indices",
        "coefficients",
        "predicted",
    )
    for key in tensor_keys:
        value = state[key]
        assert isinstance(value, torch.Tensor), f"{key} should be torch.Tensor"
        assert value.device.type == "cpu", f"{key} should stay on the configured CPU device"

    assert state["initial_trajectory"].dtype == torch.float64
    assert state["t_span"].dtype == torch.float64
    assert state["eig_full"].dtype == torch.complex128
    assert state["xi"].dtype == torch.complex128
    assert state["tensor_device"] == "cpu"
    assert state["tensor_dtype"] == "torch.float64"


def test_predict_with_diagnostics_return_tensor_preserves_tensor_contract():
    fdmd = _build_fdmd()

    prediction, diagnostics = fdmd.predict_with_diagnostics(
        initial_trajectory=np.array([1.0, 0.95, 0.9, 0.85], dtype=float),
        t_span=np.array([2.0, 2.5, 3.0], dtype=float),
        stability_threshold=0.01,
        prediction_mode_selection_policy="adaptive_tail_energy",
        max_prediction_modes=2,
        min_prediction_modes=1,
        return_tensor=True,
    )

    assert isinstance(prediction, torch.Tensor)
    assert prediction.device.type == "cpu"
    assert prediction.dtype == torch.float64
    assert prediction.shape == (3, 1)
    assert isinstance(fdmd.initial_coefficients_, torch.Tensor)
    assert fdmd.initial_coefficients_.device.type == "cpu"
    assert fdmd.initial_coefficients_.dtype == torch.complex128
    assert diagnostics["tensor_device"] == "cpu"
    assert diagnostics["tensor_dtype"] == "torch.float64"
    assert diagnostics["n_selected_prediction_modes"] == 2


def test_prepare_prediction_state_accepts_transposed_modes_tensor():
    fdmd = _build_fdmd()
    fdmd.modes_ = np.array([[1.0, 1.0]], dtype=float)

    state = fdmd._prepare_prediction_state(
        initial_trajectory=np.array([1.0, 0.9, 0.8, 0.75], dtype=float),
        t_span=np.array([2.0, 2.5], dtype=float),
        stability_threshold=0.01,
        prediction_mode_selection_policy="all_stable",
        min_prediction_modes=1,
    )

    assert state["eig_full"].shape == (2,)
    assert state["stable_mask"].shape == (2,)
    assert state["xi"].shape == (2, 1)
    assert fdmd.modes_.shape == (2, 1)
