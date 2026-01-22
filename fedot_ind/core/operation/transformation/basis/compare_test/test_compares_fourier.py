import numpy as np
import torch

from fedot_ind.core.operation.transformation.basis.fourier import FourierBasisImplementationTorch, FourierBasisImplementation
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


ESTIMATORS = {
    # "non_parametric": {"window": "hann",
    #               "detrend": False,
    #               "scale_by_freq": False,
    #               "NFFT": None},
    # "eigen": {'IP': 5,
    #           'NFFT': None,
    #           #"scale_by_freq": False,
    #           }
}
ESTIMATORS = ["non_parametric", "eigen"]


def compare_numpy_and_torch(
    estimator_name: str,
    # estimator_params: dict,
    signal: np.ndarray,
    atol: float = 1e-8,
    rtol: float = 1e-8,
):
    print(f"\n=== Testing estimator: {estimator_name} ===")

    params = {}
        # "estimator": estimator_name,
        # "threshold": 0.9,
        # "sampling_rate": 5.0,
        # "low_rank": 5,
        # "output_format": 'spectrum',#"signal",
        # "approximation": "exact",
        # "compute_heuristic_representation": False#True,#False,
    # }

    basis_np = FourierBasisImplementation(params)
    out_np = basis_np._decompose_signal(signal)

    signal_torch = torch.tensor(signal, dtype=torch.float64)

    # params["estimator_parameters"] = estimator_params
    basis_torch = FourierBasisImplementationTorch(params)
    out_torch = basis_torch._decompose_signal(signal_torch)

    out_torch_np = out_torch.detach().cpu().numpy()
    rmse = np.power((out_np - out_torch_np), 2).mean() ** 0.5
    print(f"RMSE ({estimator_name})", rmse)
    print(f"Max abs diff ({estimator_name}):", np.max(np.abs(out_np - out_torch_np)))

    assert np.allclose(
        out_np,
        out_torch_np,
        atol=atol,
        rtol=rtol,
    ), f"Mismatch for estimator={estimator_name}"

    print("✓ signal OK")


def run_all_estimators_test():
    np.random.seed(42)
    torch.manual_seed(42)

    N = 100
    signal = np.random.randn(N)
    # compare_numpy_and_torch("non_parametric", None, signal)
    # for est, param in ESTIMATORS.items():
    for est in ESTIMATORS:
        for i in range(5):
            compare_numpy_and_torch(est, signal)

    print("\n✅ All estimators passed successfully")


if __name__ == "__main__":
    run_all_estimators_test()
