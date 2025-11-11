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


def main():
    test_series = generate_series(n=5000)
    n_list = [5000, 10000, 200000]
    res = []
    for n in n_list:
        print(f"Test on {n} points:")
        test_series = generate_series(n=5000)
        for name, data in test_series.items():
            res.append(compare_recurrence(name, data))


if __name__ == "__main__":
    main()
