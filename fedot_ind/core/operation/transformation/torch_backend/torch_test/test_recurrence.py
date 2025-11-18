import time
import numpy as np
import torch

from fedot_ind.core.operation.transformation.representation.recurrence.recurrence_extractor import RecurrenceExtractor as RecurrenceNumpy
from fedot_ind.core.operation.transformation.torch_backend.recurrence.recurrence_extractor import RecurrenceExtractor as RecurrenceTorch


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

    start = time.time()
    rec_np = RecurrenceNumpy(params={"rec_metric": "cosine", "window_size": 50})
    features_np = rec_np.generate_recurrence_features(ts_data)
    t_np = time.time() - start

    start = time.time()
    rec_torch = RecurrenceTorch(params={"rec_metric": "cosine", "window_size": 50})
    features_torch = rec_torch.generate_recurrence_features(torch.tensor(ts_data, dtype=torch.float32))
    t_torch = time.time() - start

    print(f"Numpy: {t_np:.4f}s | Torch: {t_torch:.4f}s | Speed-up: {t_np / t_torch:.2f}x")

    f_np = np.array(features_np)
    f_torch = np.array(features_torch)
    diff = np.mean(np.abs(f_np - f_torch))
    print(f"Mean abs difference: {diff:.6f}", "\n")

    return t_np, t_torch, diff


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
    rec_np = RecurrenceNumpy(params={"rec_metric": "cosine", "window_size": 50})
    feat_np = rec_np.generate_recurrence_features(batch_np)
    t_np = time.time() - start
    # Torch
    start = time.time()
    rec_t = RecurrenceTorch(params={"rec_metric": "cosine", "window_size": 50})
    feat_torch = rec_t.generate_recurrence_features(batch_torch)
    t_torch = time.time() - start

    print(f"Numpy: {t_np:.4f}s | Torch: {t_torch:.4f}s | speed-up {t_np / t_torch:.2f}x")

    feat_np = np.array(feat_np, dtype=float)
    feat_torch = np.array(feat_torch, dtype=float)
    print("Shapes:", feat_np.shape, feat_torch.shape)

    diff = np.mean(np.abs(feat_np - feat_torch))
    print(f"Mean abs diff: {diff:.8f}")

    return diff


def main():
    test_series = generate_series(n=5000)
    n_list = [5000]
    res = []
    for n in n_list:
        print(f"Test on {n} points:")
        test_series = generate_series(n=5000)
        for name, data in test_series.items():
            res.append(compare_recurrence(name, data))

    # for batch of ts
    B = 5
    T = 5000
    batch_np = generate_multi_series(n=T, batch=B)
    compare_batch_recurrence(batch_np)


if __name__ == "__main__":
    main()
