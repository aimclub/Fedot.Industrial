import numpy as np

from fedot_ind.core.kernel_learning import KernelMatrixBuilder


def test_kernel_matrix_builder_returns_psd_train_and_cross_kernel():
    train = np.array([[0.0], [1.0], [2.0]])
    test = np.array([[1.5], [3.0]])

    builder = KernelMatrixBuilder(kernel="rbf", normalize="trace", psd_correction="clip")
    bundle = builder.fit_transform(train, name="rbf")
    cross = builder.transform(test)

    assert bundle.train_kernel.shape == (3, 3)
    assert cross.shape == (2, 3)
    assert np.allclose(bundle.train_kernel, bundle.train_kernel.T)
    assert np.isclose(np.trace(bundle.train_kernel), 3.0)
    assert bundle.is_psd is True
    assert bundle.diagnostics["min_eigenvalue"] >= -1e-8


def test_kernel_matrix_builder_clips_non_psd_matrix():
    builder = KernelMatrixBuilder(psd_correction="clip")
    corrected, diagnostics = builder._validate_and_correct_psd(
        np.array(
            [
                [1.0, 2.0],
                [2.0, 1.0],
            ]
        )
    )

    assert diagnostics["psd_correction"] == "clip"
    assert diagnostics["is_psd"] is True
    assert np.min(np.linalg.eigvalsh(corrected)) >= -1e-8
