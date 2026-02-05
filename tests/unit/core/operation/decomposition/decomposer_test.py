import pytest
import numpy as np
import torch

from fedot_ind.core.operation.decomposition.matrix_decomposition.decomposer import MatrixDecomposerTorch, MatrixDecomposer


@pytest.fixture
def sample_matrix():
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal((100, 100))


@pytest.fixture
def decomposition_params():
    return {
        "decomposition_type": "svd",
        "decomposition_params": {
            "spectrum_regularization": "hard_thresholding"
        },
        "min_components_number": None,
    }


def test_matrix_decomposition_numpy_vs_torch_consistency(
    sample_matrix: np.ndarray, decomposition_params: dict
) -> None:
    """
    Test the consistency of matrix decomposition results between NumPy and
    PyTorch implementations.

    This function compares the results of matrix decomposition between
    NumPy-based and PyTorch-based implementations to ensure they produce
    equivalent results within specified tolerances. It verifies that the rank,
    left eigenvectors, spectrum, and right eigenvectors match between the two
    implementations.

    Args:
        sample_matrix (np.ndarray): Input matrix to decompose. Expected to be a
        2D array.
        decomposition_params (dict): Parameters for the matrix decomposition.
            Expected keys may include 'decomposition_type',
            'decomposition_params', etc.

    Raises:
        AssertionError: If the decomposition results between NumPy and PyTorch
        implementations differ beyond the specified tolerances, or if the ranks
        do not match.

    Notes:
        - The function uses absolute tolerance (atol) of 1e-5 and relative
        tolerance (rtol) of 0.0 for numerical comparisons.
        - PyTorch tensors are moved to CPU and converted to NumPy arrays for
        comparison.
    """
    decomposer_np = MatrixDecomposer(decomposition_params)
    decomposer_torch = MatrixDecomposerTorch(decomposition_params)

    result_np = decomposer_np.apply(sample_matrix)
    result_torch = decomposer_torch.apply(
        torch.tensor(sample_matrix, dtype=torch.float64)
    )

    assert result_np["rank"] == result_torch["rank"]
    np.testing.assert_allclose(
        result_np["left_eigenvectors"],
        result_torch["left_eigenvectors"].detach().cpu().numpy(),
        rtol=0.0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        result_np["spectrum"],
        result_torch["spectrum"].detach().cpu().numpy(),
        rtol=0.0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        result_np["right_eigenvectors"],
        result_torch["right_eigenvectors"].detach().cpu().numpy(),
        rtol=0.0,
        atol=1e-5,
    )
