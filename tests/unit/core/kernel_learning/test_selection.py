import numpy as np

from fedot_ind.core.kernel_learning.contracts import KernelBundle
from fedot_ind.core.kernel_learning.selection import SparseMKLSelector, TargetKernelBuilder


def test_sparse_mkl_selector_prefers_informative_kernel():
    y = np.array([0, 0, 1, 1])
    informative = (y[:, None] == y[None, :]).astype(float)
    noise = np.eye(4)
    bundles = [
        KernelBundle(
            name="informative",
            train_kernel=informative,
            complexity={"kernel_complexity": 1.0},
        ),
        KernelBundle(
            name="noise",
            train_kernel=noise,
            complexity={"kernel_complexity": 1.0},
        ),
    ]

    report = SparseMKLSelector(
        complexity_penalty=0.0,
        redundancy_penalty=0.0,
        min_weight=0.01,
    ).fit(bundles, y, task_type="classification")

    weights = dict(zip(report.generator_names, report.weights))
    assert weights["informative"] > weights["noise"]
    assert "informative" in report.selected_generators


def test_sparse_mkl_selector_uses_uniform_weights_when_scores_are_zero():
    y = np.array([0, 1])
    zero = np.zeros((2, 2))
    bundles = [
        KernelBundle(name="a", train_kernel=zero, complexity={"kernel_complexity": 0.0}),
        KernelBundle(name="b", train_kernel=zero, complexity={"kernel_complexity": 0.0}),
    ]

    report = SparseMKLSelector().fit(bundles, y, task_type="classification")

    assert report.weights == (0.5, 0.5)
    assert report.selected_generators == ("a", "b")


def test_regression_target_kernel_is_square_and_finite():
    target_kernel = TargetKernelBuilder(task_type="regression").build(np.array([1.0, 2.0, 4.0]))

    assert target_kernel.shape == (3, 3)
    assert np.all(np.isfinite(target_kernel))
    assert np.allclose(np.diag(target_kernel), 1.0)
