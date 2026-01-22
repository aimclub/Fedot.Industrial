import dask
import pytest
import torch
import os
import shutil
from fedot.core.data.data import OutputData

from fedot_ind.core.operation.dummy.dummy_operation import init_input_data
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.transformation.basis.fourier import FourierBasisImplementation, FourierBasisImplementationTorch
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator

CACHE_PATH = "/workspaces/Fedot.Industrial/cache"
torch.manual_seed(12345)
np.random.seed(12345)

@pytest.fixture
def dataset():
    (X_train, y_train), (X_test, y_test) = TimeSeriesDatasetsGenerator(
        num_samples=20, max_ts_len=50, binary=True, test_size=0.5).generate_data()
    return X_train, y_train, X_test, y_test

def clear_folder_simple(folder_path: str) -> None:
    """
    Простая очистка папки (удаляет все файлы и подпапки).
    Использует только стандартную библиотеку.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Не удалось удалить {file_path}: {e}')


# @pytest.fixture
def input_train():
    x_train = np.random.rand(100, 1, 100)
    y_train = np.random.rand(100).reshape(-1, 1)
    input_train_data = init_input_data(x_train, y_train)
    return input_train_data


def test_transform(input_train, atol=1e-6, rtol=0):
    basis = FourierBasisImplementation({})
    train_features_np = basis.transform(input_data=input_train, use_cache=False)

    basis_torch = FourierBasisImplementationTorch({})
    input_train.features = torch.tensor(input_train.features, 
                                        dtype=torch.float64)
    train_features_torch = basis_torch.transform(input_data=input_train, use_cache=False)
    out_np = np.array(train_features_np.predict)
    out_torch_np = np.array(train_features_torch.predict)
    # out_np = train_features_np.predict
    # out_torch_np = train_features_torch.predict
    rmse = np.power((out_np - out_torch_np), 2).mean() ** 0.5
    print(f"RMSE:", rmse)
    print(f"Max abs diff:", np.max(np.abs(out_np - out_torch_np)))
    # print(out_torch_np.shape)
    # print(out_np[1])
    # print(out_torch_np[1])
    # assert np.allclose(
    #     out_np,
    #     out_torch_np,
    #     atol=atol,
    #     rtol=rtol,
    # ), f"Mismatch for numpy and torch versions of estimators"
    assert isinstance(train_features_torch, OutputData)
    assert train_features_torch.features.shape[0] == input_train.features.shape[0]


def test_transform_one_sample_test(input_train, atol=1e-6, rtol=.0):
    # sample = input_train.features[0]
    k = 1
    for sample in input_train.features:
        # print(sample.shape)
        k+=1
        basis_np = FourierBasisImplementation({})
        transformed_sample_np = basis_np._transform_one_sample(sample)
        transformed_sample_np = dask.compute(transformed_sample_np)[0]

        basis_torch = FourierBasisImplementationTorch({})
        sample_torch = torch.tensor(sample, dtype=torch.float64)
        transformed_sample_torch = basis_torch._transform_one_sample(sample_torch)
        transformed_sample_torch = dask.compute(transformed_sample_torch)[0]
        out_np = np.array(transformed_sample_np)
        out_torch_np = np.array(transformed_sample_torch)
        rmse = np.power((out_np - out_torch_np), 2).mean() ** 0.5
        if rmse>1e-6:
            print(f"{k}RMSE:", rmse)
            print(f"Max abs diff:", np.max(np.abs(out_np - out_torch_np)))
        # assert np.allclose(
        #     np.array(transformed_sample_np),
        #     np.array(transformed_sample_torch),
        #     atol=atol,
        #     rtol=rtol,
        # ), f"Mismatch for numpy and torch versions of estimators"
        # assert isinstance(transformed_sample_torch, list)
        # assert transformed_sample_torch[0].shape[0] == len(sample)


def test_transform_one_sample(input_train, atol=1e-6, rtol=.0):
    sample = input_train.features[0]
    basis_np = FourierBasisImplementation({})
    transformed_sample_np = basis_np._transform_one_sample(sample)
    transformed_sample_np = dask.compute(transformed_sample_np)[0]

    basis_torch = FourierBasisImplementationTorch({})
    sample_torch = torch.tensor(sample, dtype=torch.float64)
    transformed_sample_torch = basis_torch._transform_one_sample(sample_torch)
    transformed_sample_torch = dask.compute(transformed_sample_torch)[0]
    out_np = np.array(transformed_sample_np)
    out_torch_np = np.array(transformed_sample_torch)
    rmse = np.power((out_np - out_torch_np), 2).mean() ** 0.5
    print(f"RMSE:", rmse)
    print(f"Max abs diff:", np.max(np.abs(out_np - out_torch_np)))
    assert np.allclose(
        np.array(transformed_sample_np),
        np.array(transformed_sample_torch),
        atol=atol,
        rtol=rtol,
    ), f"Mismatch for numpy and torch versions of estimators"
    assert isinstance(transformed_sample_torch, list)
    assert transformed_sample_torch[0].shape[0] == len(sample)


def test_decompose_signal(input_train, atol=1e-6, rtol=.0):
    basis_np = FourierBasisImplementation({})
    sample = input_train.features[0].reshape(-1)
    transformed_sample_np = basis_np._decompose_signal(sample)
    basis_torch = FourierBasisImplementationTorch({})
    sample = torch.tensor(sample, dtype=torch.float64)
    transformed_sample_torch = basis_torch._decompose_signal(sample)
    transformed_sample_torch = transformed_sample_torch.detach().cpu().numpy()
    
    rmse = np.power((transformed_sample_np - 
                     transformed_sample_torch), 2).mean() ** 0.5
    print(f"RMSE:", rmse)
    print(f"Max abs diff:", np.max(np.abs(transformed_sample_np - 
                                          transformed_sample_torch)))
    assert np.allclose(
        transformed_sample_np,
        transformed_sample_torch,
        atol=atol,
        rtol=rtol,
    ), f"Mismatch for numpy and torch versions of estimators"
    assert isinstance(transformed_sample_torch, np.ndarray)
    assert transformed_sample_torch.shape[-1] == sample.shape[-1]


if __name__ == "__main__":
    clear_folder_simple(CACHE_PATH)
    # test_transform(input_train=input_train())
    # test_decompose_signal(input_train=input_train())
    test_transform_one_sample(input_train=input_train())