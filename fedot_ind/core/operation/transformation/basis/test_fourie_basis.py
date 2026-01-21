import dask
import pytest
import torch
from fedot.core.data.data import OutputData

from fedot_ind.core.operation.dummy.dummy_operation import init_input_data
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.transformation.basis.fourier import FourierBasisImplementation, FourierBasisImplementationTorch
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


@pytest.fixture
def dataset():
    (X_train, y_train), (X_test, y_test) = TimeSeriesDatasetsGenerator(
        num_samples=20, max_ts_len=50, binary=True, test_size=0.5).generate_data()
    return X_train, y_train, X_test, y_test


# @pytest.fixture
# def input_train(dataset):
#     X_train, y_train, X_test, y_test = dataset
#     input_train_data = init_input_data(X_train, y_train)
#     return input_train_data

# @pytest.fixture
def input_train():
    x_train = np.random.rand(100, 1, 100)
    y_train = np.random.rand(100).reshape(-1, 1)
    input_train_data = init_input_data(x_train, y_train)
    return input_train_data


def test_transform(input_train, atol=1e-6, rtol=0):
    basis = FourierBasisImplementation({})
    train_features = basis.transform(input_data=input_train)
    basis_torch = FourierBasisImplementationTorch({})
    input_train.features = torch.Tensor(input_train.features)
    print(type(input_train.features))
    train_features_torch = basis_torch.transform(input_data=input_train)
    out_np = train_features.features
    out_torch_np = train_features_torch.features.detach().cpu().numpy()
    rmse = np.power((out_np - out_torch_np), 2).mean() ** 0.5
    print(f"RMSE:", rmse)
    print(f"Max abs diff:", np.max(np.abs(out_np - out_torch_np)))
    assert np.allclose(
        out_np,
        out_torch_np,
        atol=atol,
        rtol=rtol,
    ), f"Mismatch for numpy and torch versions of estimators"
    assert isinstance(train_features_torch, OutputData)
    assert train_features_torch.features.shape[0] == input_train.features.shape[0]


if __name__ == "__main__":
    test_transform(input_train=input_train())