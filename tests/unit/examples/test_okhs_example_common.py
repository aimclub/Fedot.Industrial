import numpy as np
import pytest

from examples.rkhs_okhs.example_common import MittagLefflerKernel, RBFKernel, generate_trajectories_pycaputo


def test_rbf_kernel_is_symmetric_for_same_inputs():
    kernel = RBFKernel(gamma=0.5)
    left = np.array([1.0, 2.0])
    right = np.array([-1.0, 0.5])

    assert kernel._compute_single_kernel(left, right) == kernel._compute_single_kernel(right, left)


def test_mittag_leffler_kernel_validates_q():
    with pytest.raises(ValueError):
        MittagLefflerKernel(q=0.0)


def test_generate_trajectories_pycaputo_raises_without_dependency(monkeypatch):
    monkeypatch.setattr("examples.rkhs_okhs.example_common.make_fixed_controller", None)

    with pytest.raises(ImportError):
        generate_trajectories_pycaputo(
            lambda t, y: y,
            q_true=0.8,
            n_trajectories=1,
            n_steps=5,
            T_max=1.0,
            dim=1,
        )
