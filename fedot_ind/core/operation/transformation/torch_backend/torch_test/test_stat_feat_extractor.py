import torch
import numpy as np
import time
import itertools
from fedot_ind.core.operation.transformation.representation.statistical.quantile_extractor import QuantileExtractor
from fedot_ind.core.operation.transformation.torch_backend.statistical.quantile_extractor import TorchQuantileExtractor
from fedot.core.operations.operation_parameters import OperationParameters


def test_stat_extractor(shape_array, window_size=10, stride=2, add_global_features=True):
    params = OperationParameters(
        window_size=window_size,
        stride=stride,
        add_global_features=add_global_features
    )
    numpy_extractor = QuantileExtractor(params)
    torch_extractor = TorchQuantileExtractor(params)

    x_np = np.random.randn(shape_array[0], shape_array[1]).astype(np.float32).squeeze()
    # x_np = np.random.randn(256).astype(np.float32)
    x_torch = torch.tensor(x_np, dtype=torch.float32)

    # print('numpy')
    start_np = time.perf_counter()
    np_result = numpy_extractor.generate_features_from_ts(x_np)
    t_np = time.perf_counter() - start_np
    # print(np_result.shape, "\n")
    # print('torch')
    start_torch = time.perf_counter()
    torch_result = torch_extractor.generate_features_from_ts(x_torch)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_torch = time.perf_counter() - start_torch
    # print(torch_result.shape)
    if isinstance(torch_result, torch.Tensor):
        torch_result_np = torch_result.detach().cpu().numpy()
    else:
        torch_result_np = np.array(torch_result)

    diff = np.abs(np_result - torch_result_np).mean()
    mse = np.power((np_result - torch_result_np), 2).mean() ** 0.5

    print(f"MAE: {diff:.6f}")
    print(f"RMSE {mse:.6f}")
    print(f"NumPy время: {t_np:.6f} сек")
    print(f"Torch время: {t_torch:.6f} сек")
    print(f"Ускорение: {t_np / t_torch:.2f}x \n")


def main():
    test_shapes = [[1, 256], [6, 512], [8, 1024]]  # [[6, 512]]
    strides = [1, 2, 3]
    glob_feat = [True, False]
    params_combinations = itertools.product(test_shapes, strides, glob_feat)
    for shape, stride, global_feat in params_combinations:
        print(f"Test with shape = {shape}, stride = {stride}, global_feat = {global_feat}")
        test_stat_extractor(
            shape_array=shape,
            stride=stride,
            add_global_features=global_feat
        )


if __name__ == "__main__":
    main()
