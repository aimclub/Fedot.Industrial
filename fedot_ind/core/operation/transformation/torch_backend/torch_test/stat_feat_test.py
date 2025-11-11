import torch
import numpy as np
import pandas as pd
import time
import inspect
import importlib.util
import contextlib
import io

torch_path = "/workspaces/Fedot.Industrial/fedot_ind/core/operation/transformation/torch_backend/statistical/stat_features.py"
numpy_path = "/workspaces/Fedot.Industrial/fedot_ind/core/operation/transformation/representation/statistical/stat_features.py"


def import_module_from_path(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


torch_mod = import_module_from_path(torch_path, "stat_features_torch")
numpy_mod = import_module_from_path(numpy_path, "stat_features_numpy")


def time_call(func, *args, runs=30):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        start = time.perf_counter()
        for _ in range(runs):
            _ = func(*args)
        torch.cuda.synchronize() if torch.cuda.is_available() and isinstance(args[0], torch.Tensor) else None
        return (time.perf_counter() - start) / runs


def benchmark_all(runs=30):
    torch_funcs = {f.__name__: f for f in vars(torch_mod).values() if inspect.isfunction(f)}
    numpy_funcs = {f.__name__: f for f in vars(numpy_mod).values() if inspect.isfunction(f)}
    common_funcs = {n: (torch_funcs[n], numpy_funcs[n]) for n in torch_funcs if n in numpy_funcs}

    results = []

    for name, (torch_func, numpy_func) in common_funcs.items():
        try:
            x_np_single = np.random.randn(10000).astype(np.float32)
            x_torch_single_cpu = torch.tensor(x_np_single, device="cpu")
            x_torch_single_gpu = torch.tensor(x_np_single, device="cuda") if torch.cuda.is_available() else None

            x_np_batch = np.random.randn(32, 2048).astype(np.float32)
            x_torch_batch_cpu = torch.tensor(x_np_batch, device="cpu")
            x_torch_batch_gpu = torch.tensor(x_np_batch, device="cuda") if torch.cuda.is_available() else None

            t_np_cpu = time_call(numpy_func, x_np_single, runs=runs)
            t_torch_cpu = time_call(torch_func, x_torch_single_cpu, runs=runs)
            t_torch_gpu = time_call(torch_func, x_torch_single_gpu, runs=runs) if x_torch_single_gpu is not None else None
            t_torch_cpu_batch = time_call(torch_func, x_torch_batch_cpu, runs=runs)
            t_torch_gpu_batch = (
                time_call(torch_func, x_torch_batch_gpu, runs=runs)
                if x_torch_batch_gpu is not None
                else None
            )

            try:
                torch_res_cpu = torch_func(x_torch_single_cpu)
                numpy_res = numpy_func(x_np_single)
                if isinstance(torch_res_cpu, torch.Tensor):
                    torch_res_cpu = torch_res_cpu.detach().cpu().numpy()
                match = np.allclose(torch_res_cpu, numpy_res, rtol=1e-4, atol=1e-4, equal_nan=True)
            except Exception as e:
                match = f"ERROR: {e}"

            results.append(
                (name, t_np_cpu, t_torch_cpu, t_torch_gpu, t_torch_cpu_batch, t_torch_gpu_batch, match)
            )
        except Exception as e:
            results.append((name, "ERROR", "ERROR", None, None, None, str(e)))

    print(f"\n{'Function':30} {'t_np_cpu':>10} {'t_torch_cpu':>13} {'t_torch_gpu':>13} {'t_torch_cpu_batch':>18} {'t_torch_gpu_batch':>18} {'Match?':>10}")
    print("-" * 100)
    for name, t_np, t_cpu, t_gpu, t_cpu_b, t_gpu_b, match in results:
        def fmt(x): return f"{x:.6f}" if isinstance(x, (int, float)) else ("N/A" if x is None else str(x))
        print(f"{name:30} {fmt(t_np):>10} {fmt(t_cpu):>13} {fmt(t_gpu):>13} {fmt(t_cpu_b):>18} {fmt(t_gpu_b):>18}")

    return results


if __name__ == "__main__":
    print("🚀 Running benchmark for statistical features...\n")
    results = benchmark_all(runs=20)
    df = pd.DataFrame(results, columns=[
        "function", "t_np_cpu", "t_torch_cpu", "t_torch_gpu", "t_torch_cpu_batch", "t_torch_gpu_batch", "match"
    ])
    import os

    save_path = "/workspaces/Fedot.Industrial/fedot_ind/core/operation/transformation/torch_backend/statistical/benchmark_results.xlsx"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        df.to_excel(save_path, index=False, engine="openpyxl")
    except Exception as e:
        print(f"{e}")

