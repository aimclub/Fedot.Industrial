import torch
import pandas as pd
import numpy as np
from tabulate import tabulate
import time
import itertools
from fedot_ind.core.operation.transformation.representation.statistical.quantile_extractor import QuantileExtractor
from fedot_ind.core.operation.transformation.torch_backend.statistical.quantile_extractor import TorchQuantileExtractor
from fedot.core.operations.operation_parameters import OperationParameters
from tqdm import tqdm


@torch.no_grad()
def warm_up_cuda_computations(n_iters=5, size=2048, device=None):
    """ Function for CUDA warming. It is used before time measuring.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)
    for _ in tqdm(range(n_iters)):
        C = A @ B
        C = torch.sin(C) * torch.exp(C)
        _ = C.sum()
    if device == "cuda":
        torch.cuda.synchronize()
    print("Warm-up done")


def test_stat_extractor(shape_array, window_size=10, stride=2, add_global_features=True):
    """ Function of measuring time of statictical features extracting for previous realisation (numpy)
    and new realisation (torch) on CPU and GPU. First, create random time series with shape shape_array,
    then extract features and measure time. Time for GPU is measured 10 times, then average value is taking.

    Returns:
        dict:
            shape of data: shape of data to create
            stride: stride for extractors
            add_global_features: parameter for extractors
            numpy CPU time (sec): time in seconds of extracting features through numpy realisation on CPU
            torch CPU time (sec): time in seconds of extracting features through torch realisation on CPU
            speedup: the value of deviding numpy CPU time by torch CPU time
            AVG torch GPU time (sec): average time in seconds of extracting features through torch realisation on GPU
            speedup GPU: the value of deviding numpy CPU time by AVG torch GPU time
            RMSE: root mean squared error between numpy and torch realisations
    """
    params = OperationParameters(
        window_size=window_size,
        stride=stride,
        add_global_features=add_global_features
    )
    numpy_extractor = QuantileExtractor(params)
    torch_extractor = TorchQuantileExtractor(params)

    x_np = np.random.randn(shape_array[0], shape_array[1]).astype(np.float32).squeeze()
    x_torch = torch.tensor(x_np, dtype=torch.float32)

    # numpy CPU
    start_np = time.perf_counter()
    np_result = numpy_extractor.generate_features_from_ts(x_np)
    t_np = time.perf_counter() - start_np

    # torch CPU
    start_torch = time.perf_counter()
    torch_result = torch_extractor.generate_features_from_ts(x_torch)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_torch = time.perf_counter() - start_torch
    torch_result.detach().cpu().numpy()

    # torch GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Working on GPU")
    warm_up_cuda_computations(device=device)
    times_gpu = []
    x_gpu = x_torch.clone().to(device)
    for i in range(10):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        torch_result_gpu = torch_extractor.generate_features_from_ts(x_gpu)
        end_event.record()
        torch.cuda.synchronize()
        t_torch_gpu = start_event.elapsed_time(end_event) / 1000
        times_gpu.append(t_torch_gpu)
    avg_time_gpu = np.mean(np.array(times_gpu))
    torch_result_gpu_np = torch_result_gpu.detach().cpu().numpy()

    rmse = np.power((np_result - torch_result_gpu_np), 2).mean() ** 0.5
    return {
        "shape of data": tuple(shape_array),
        "stride": stride,
        "add_global_features": add_global_features,
        "numpy CPU time (sec)": t_np,
        "torch CPU time (sec)": t_torch,
        "speedup": round(t_np / t_torch, 2),
        "AVG torch GPU time (sec)": avg_time_gpu,
        "speedup GPU": round(t_np / avg_time_gpu, 2),
        "RMSE": rmse,
    }


def main():
    test_shapes = [[1, 1000], [6, 10000], [30, 10000]]
    windows = [1, 10]
    strides = [1, 2]
    glob_feat = [True, False]
    results = []
    params_combinations = itertools.product(test_shapes, strides, glob_feat, windows)
    for shape, stride, global_feat, w in params_combinations:
        print(f"Test with shape = {shape}, stride = {stride}, global_feat = {global_feat} \n")
        res = test_stat_extractor(
            shape_array=shape,
            stride=stride,
            add_global_features=global_feat,
            window_size=w
        )
        results.append(res)

    df = pd.DataFrame(results)
    path = ""
    df.to_csv(path + 'stat_extractor.csv', index=False)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=True))


if __name__ == "__main__":
    main()
