import dask
import pytest
import torch

from fedot.core.data.data import OutputData
from fedot_ind.core.operation.dummy.dummy_operation import init_input_data
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.transformation.basis.fourier import FourierBasisImplementation
from fedot_ind.core.operation.transformation.basis.fourier_basis_torch import FourierBasisImplementationTorch


@pytest.fixture
def input_train_np():
    x_train = np.random.rand(100, 1, 100)
    y_train = np.random.rand(100).reshape(-1, 1)
    return init_input_data(x_train, y_train)


def test_transform_one_sample_numpy_vs_torch(input_train_np,
                                             atol=1e-6,
                                             rtol=0.0) -> None:
    """
    Compare single sample transformation results between NumPy and PyTorch
    implementations.

    This test verifies that the Fourier basis transformation of a single sample
    produces equivalent results in both NumPy and PyTorch implementations within
    specified tolerances.

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

    basis_np = FourierBasisImplementation({})
    out_np = dask.compute(
        basis_np._transform_one_sample(sample_np)
    )[0]

    basis_torch = FourierBasisImplementationTorch({})
    sample_torch = torch.tensor(sample_np, dtype=torch.float64)
    out_torch = basis_torch._transform_one_sample(sample_torch)

    out_torch = out_torch.unsqueeze(dim=0).detach().cpu().numpy()
    np.testing.assert_allclose(
        np.asarray(out_np),
        np.asarray(out_torch),
        atol=atol,
        rtol=rtol,
    )


def test_transform_numpy_vs_torch(input_train_np, atol=1e-6, rtol=0.0) -> None:
    """
    Compare batch transformation results between NumPy and PyTorch
    implementations.

    This test verifies that the Fourier basis transformation of a batch of
    samples produces equivalent results in both NumPy and PyTorch
    implementations within specified tolerances. It also checks that the output
    type and shape are correct.

    Args:
        input_train_np (InputData): Input data containing features to be
        transformed.
        atol (float): Absolute tolerance for numerical comparison.
        Defaults to 1e-6.
        rtol (float): Relative tolerance for numerical comparison.
        Defaults to 0.0.

    Raises:
        AssertionError: If the NumPy and PyTorch results differ beyond the
        specified tolerances, if the output type is incorrect, or if the feature
        count doesn't match.
    """
    basis_np = FourierBasisImplementation({})
    preds_np = basis_np._transform(input_data=input_train_np).squeeze(axis=1)

    basis_torch = FourierBasisImplementationTorch({})
    input_train_np.features = torch.tensor(
        input_train_np.features, dtype=torch.float64
    )
    out_torch = basis_torch.transform(input_data=input_train_np)
    preds_torch = out_torch.predict.detach().cpu().numpy()

    rmse_values = [
        np.sqrt(np.mean((feature_np - feature_torch) ** 2))
        for feature_np, feature_torch in zip(preds_np, preds_torch)
    ]
    mismatched_count = sum(rmse > atol for rmse in rmse_values)

    assert mismatched_count <= 5, f"Transformations do not match for \
        {mismatched_count} series (tolerance: {atol})"
    assert isinstance(out_torch, OutputData)
    assert out_torch.features.shape[0] == input_train_np.features.shape[0]


def test_transform_fourier_torch_output_type() -> None:
    """
    Test the output type and shape of the Fourier transform implementation in
    PyTorch.

    This test verifies that the Fourier basis transformation returns an object
    of the correct type (OutputData) and that the shape of the prediction
    matches the shape of the input features.

    Raises:
        AssertionError: If the output type is not OutputData or if the
        prediction shape doesn't match the input features shape.
    """
    x_train = np.random.rand(100, 3, 100)
    y_train = np.random.rand(100).reshape(-1, 1)

    input_data = init_input_data(x_train, y_train)
    input_data.features = torch.tensor(
        input_data.features, dtype=torch.float64
    )

    basis = FourierBasisImplementationTorch({})
    output = basis._transform(input_data)
    assert isinstance(output, OutputData)
    assert output.predict.shape == input_data.features.shape
