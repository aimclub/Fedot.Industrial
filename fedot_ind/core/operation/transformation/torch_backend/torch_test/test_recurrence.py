import time
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate

from fedot_ind.core.operation.transformation.representation.recurrence.recurrence_extractor import RecurrenceExtractor as RecurrenceNumpy
from fedot_ind.core.operation.transformation.torch_backend.recurrence.recurrence_extractor import RecurrenceExtractor as RecurrenceTorch


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
    print("Warm-up done", "\n")


def generate_series(n=1000):
    t = np.linspace(0, 10 * np.pi, n)
    sinus = np.sin(t)
    noise = np.random.normal(0, 0.1, n)
    mixed = sinus + noise
    return {
        "sinus": sinus,
        "sinus+noise": mixed
    }


def compare_recurrence(ts_name, ts_data):
    print(f"\nTesting series: {ts_name}")
    # numpy cpu
    rec_np = RecurrenceNumpy(params={"rec_metric": "cosine", "window_size": 50, 'stride': 2})
    start = time.perf_counter()
    features_np = rec_np.generate_recurrence_features(ts_data)
    t_np = time.perf_counter() - start

    # torch cpu
    rec_torch = RecurrenceTorch(params={"rec_metric": "cosine", "window_size": 50, 'stride': 2})
    start = time.perf_counter()
    features_torch = rec_torch.generate_recurrence_features(torch.tensor(ts_data, dtype=torch.double))
    t_torch = time.perf_counter() - start

    # torch GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Working on GPU")
    rec_torch = RecurrenceTorch(params={"rec_metric": "cosine", "window_size": 50, 'stride': 2})
    warm_up_cuda_computations(device=device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    torch_result_gpu = rec_torch.generate_recurrence_features(torch.tensor(ts_data, dtype=torch.float32).to(device))
    end_event.record()
    torch.cuda.synchronize()
    t_torch_gpu = start_event.elapsed_time(end_event) / 1000

    f_np = np.array(features_np)
    f_torch = np.array(features_torch)
    rmse = np.mean(np.abs(f_np - f_torch)) ** 0.5
    return {
        "data": ts_name,
        "data shape": ts_data.shape,
        "numpy CPU time (sec)": t_np,
        "torch CPU time (sec)": t_torch,
        "speedup": round(t_np / t_torch, 2),
        "torch GPU time (sec)": t_torch_gpu,
        "speedup GPU": round(t_np / t_torch_gpu, 2),
        "RMSE": rmse,
    }


def generate_multi_series(n=1000, batch=5):
    t = np.linspace(0, 10 * np.pi, n)
    rows = [
        np.sin(t),
        np.sin(t) + np.random.normal(0, 0.1, n),
        np.sin(2 * t),
        np.cos(t),
        np.random.normal(0, 1, n)
    ]
    rows = rows[:batch]
    return np.vstack(rows)


def compare_batch_recurrence(batch_np):
    print(f"\n Testing batch of {batch_np.shape[0]} series")
    batch_torch = torch.tensor(batch_np, dtype=torch.float32)

    # NumPy
    start = time.time()
    rec_np = RecurrenceNumpy(params={"rec_metric": "cosine", "window_size": 50, 'stride': 2})
    feat_np = rec_np.generate_recurrence_features(batch_np)
    t_np = time.time() - start
    # Torch
    start = time.time()
    rec_t = RecurrenceTorch(params={"rec_metric": "cosine", "window_size": 50, 'stride': 2})
    feat_torch = rec_t.generate_recurrence_features(batch_torch)
    t_torch = time.time() - start
    # GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Working on GPU")
    rec_torch = RecurrenceTorch(params={"rec_metric": "cosine", "window_size": 50, 'stride': 2})
    warm_up_cuda_computations(device=device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    torch_result_gpu = rec_torch.generate_recurrence_features(torch.tensor(batch_torch, dtype=torch.float32).to(device))
    end_event.record()
    torch.cuda.synchronize()
    t_torch_gpu = start_event.elapsed_time(end_event) / 1000

    feat_np = np.array(feat_np, dtype=float)
    feat_torch = np.array(feat_torch, dtype=float)

    rmse = np.mean(np.abs(feat_np - feat_torch)) ** 0.5
    return {
        "data": "_",
        "data shape": batch_torch.shape,
        "numpy CPU time (sec)": t_np,
        "torch CPU time (sec)": t_torch,
        "speedup": round(t_np / t_torch, 2),
        "torch GPU time (sec)": t_torch_gpu,
        "speedup GPU": round(t_np / t_torch_gpu, 2),
        "RMSE": rmse,
    }


def main():
    n_list = [10000, 30000]
    res = []
    for n in n_list:
        print(f"Test on {n} points:")
        test_series = generate_series(n)
        batch_np = generate_multi_series(n=n, batch=5)
        for name, data in test_series.items():
            res.append(compare_recurrence(name, data))
        res.append(compare_batch_recurrence(batch_np))
    df = pd.DataFrame(res)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=True))


if __name__ == "__main__":
    main()
