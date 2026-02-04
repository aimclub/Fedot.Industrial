import dask
import pytest
import torch
from fedot.core.data.data import OutputData

from fedot_ind.core.operation.dummy.dummy_operation import init_input_data
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.transformation.basis.eigen_basis_torch import EigenBasisImplementationTorch
from fedot_ind.core.operation.transformation.basis.eigen_basis import EigenBasisImplementation


@pytest.fixture
def input_train_np():
    x_train = np.random.rand(100, 1, 100)
    y_train = np.random.rand(100).reshape(-1, 1)
    return init_input_data(x_train, y_train)


def test_eigen_transform_torch_output_type(input_train_np) -> None:
    """
    Test the output type of the EigenBasisImplementationTorch transformation.

    This test verifies that the transformation method of 
    EigenBasisImplementationTorch returns an object of the correct type
    (OutputData) when processing input data.

    Raises:
        AssertionError: If the output is not an instance of OutputData.
    """
    input_train_np.features = torch.tensor(
        input_train_np.features, dtype=torch.float64
    )

    basis = EigenBasisImplementationTorch({})
    output = basis._transform(input_train_np)

    assert isinstance(output, OutputData)


def test_eigen_transform_one_sample_numpy_vs_torch(
    input_train_np, atol=1e-6, rtol=0.0
) -> None:
    """
    Compare single sample transformation results between NumPy and PyTorch
    implementations for EigenBasis.

    This test verifies that the transformation of a single sample produces
    equivalent results in both NumPy and PyTorch implementations of EigenBasis
    within specified tolerances.

    Args:
        input_train_np (InputData): Input data containing features to be
        transformed.
        atol (float): Absolute tolerance for numerical comparison.
        Defaults to 1e-6.
        rtol (float): Relative tolerance for numerical comparison.
        Defaults to 0.0.

    Raises:
        AssertionError: If the NumPy and PyTorch results differ beyond the
        specified tolerances.
    """
    sample_np = input_train_np.features[0]

    basis_np = EigenBasisImplementation({})
    out_np = dask.compute(
        basis_np._transform_one_sample(sample_np)
    )[0]

    basis_torch = EigenBasisImplementationTorch({})
    out_torch = dask.compute(
        basis_torch._transform_one_sample(
            torch.tensor(sample_np, dtype=torch.float64)
        )
    )[0]

    np.testing.assert_allclose(
        np.asarray(out_np),
        np.asarray(out_torch),
        atol=atol,
        rtol=rtol,
    )


def test_eigen_transform_numpy_vs_torch(
    input_train_np, atol=1e-6, rtol=0.0
) -> None:
    """
    Compare batch transformation results between NumPy and PyTorch
    implementations for EigenBasis.

    This test verifies that the transformation of a batch of samples produces
    equivalent results in both NumPy and PyTorch implementations of EigenBasis
    within specified tolerances. It also checks that the output type and shape
    are correct.

    Args:
        input_train_np (InputData): Input data containing features to be
        transformed.
        atol (float): Absolute tolerance for numerical comparison.
        Defaults to1e-6.
        rtol (float): Relative tolerance for numerical comparison.
        Defaults to 0.0.

    Raises:
        AssertionError: If the NumPy and PyTorch results differ beyond the
        specified tolerances, if the output type is incorrect, or if the feature
        count doesn't match.
    """
    basis_np = EigenBasisImplementation({})
    out_np = basis_np.transform(input_data=input_train_np)

    basis_torch = EigenBasisImplementationTorch({})
    input_train_np.features = torch.tensor(
        input_train_np.features, dtype=torch.float64
    )
    out_torch = basis_torch.transform(input_data=input_train_np)

    np.testing.assert_allclose(
        np.asarray(out_np.predict),
        np.asarray(out_torch.predict),
        atol=atol,
        rtol=rtol,
    )

    assert isinstance(out_torch, OutputData)
    assert out_torch.features.shape[0] == input_train_np.features.shape[0]
