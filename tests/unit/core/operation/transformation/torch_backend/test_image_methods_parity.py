"""Optional numerical parity tests against pyts (GAF/MTF) and scipy (STFT).

These tests are skipped automatically when ``pyts`` or ``scipy`` are not
installed. They are intentionally **not** listed in project requirements.
"""

from __future__ import annotations
from fedot_ind.core.operation.transformation.torch_backend.image.stft_transformation import (
    STFTSpectrogram,
)
from fedot_ind.core.operation.transformation.torch_backend.image.mtf_transformation import MTF
from fedot_ind.core.operation.transformation.torch_backend.image.gaf_transformation import GAF

import numpy as np
import pytest
import torch

pytest.importorskip("torch")


RMSE_TOL_PYTS = 1e-4
RMSE_TOL_SCIPY = 1e-4

GAF_BENCHMARK_PARAMS = {
    "overlapping": True,
    "image_size": 0.7,
    "torch_device": "cpu",
}

MTF_BENCHMARK_PARAMS = {
    "image_size": 0.7,
    "n_bins": 8,
    "strategy": "quantile",
    "overlapping": True,
    "flatten": False,
    "torch_device": "cpu",
}

MTF_STRATEGY_CASES = [
    pytest.param(
        "uniform",
        {"image_size": 0.7, "n_bins": 8, "overlapping": True, "flatten": False},
        id="uniform-overlap",
    ),
    pytest.param(
        "quantile",
        {"image_size": 0.7, "n_bins": 8, "overlapping": True, "flatten": False},
        id="quantile-overlap",
    ),
    pytest.param(
        "normal",
        {"image_size": 0.7, "n_bins": 8, "overlapping": True, "flatten": False},
        id="normal-overlap",
    ),
    pytest.param(
        "uniform",
        {"image_size": 0.55, "n_bins": 8, "overlapping": False, "flatten": False},
        id="uniform-no-overlap",
    ),
    pytest.param(
        "quantile",
        {"image_size": 0.55, "n_bins": 8, "overlapping": False, "flatten": False},
        id="quantile-no-overlap",
    ),
    pytest.param(
        "normal",
        {"image_size": 0.55, "n_bins": 8, "overlapping": False, "flatten": False},
        id="normal-no-overlap",
    ),
]

STFT_BENCHMARK_PARAMS = {
    "window_size": 16,
    "hop_length": 4,
    "n_fft": 16,
    "window_type": "hann",
    "center": False,
    "power": 2.0,
    "normalized": False,
    "torch_device": "cpu",
}


def _rmse(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.sqrt(np.mean((reference - candidate) ** 2)))


def _scipy_stft_reference_batch(
    data: np.ndarray,
    *,
    window_size: int,
    hop_length: int,
    n_fft: int,
    power: float,
) -> np.ndarray:
    """SciPy STFT reference aligned with ``torch.stft`` (benchmark recipe).

    Requires ``n_fft == window_size``, ``center=False``, ``normalized=False``.
    SciPy ``scaling='spectrum'`` divides by ``sum(window)``; multiply back so
    amplitudes match the non-normalized torch implementation.
    """
    pytest.importorskip("scipy")
    from scipy.signal import stft as scipy_stft, windows

    if n_fft != window_size:
        raise ValueError("SciPy parity reference requires n_fft == window_size.")

    window = windows.hann(window_size, sym=False)
    window_sum = np.sum(window)
    noverlap = window_size - hop_length

    references = []
    for row in data:
        _, _, stft_matrix = scipy_stft(
            row,
            nperseg=window_size,
            noverlap=noverlap,
            nfft=n_fft,
            window=window,
            boundary=None,
            padded=False,
            return_onesided=True,
            scaling="spectrum",
        )
        references.append((np.abs(stft_matrix) * window_sum) ** power)
    return np.stack(references, axis=0)


@pytest.fixture
def pyts_benchmark_data():
    rng = np.random.default_rng(42)
    return rng.normal(size=(100, 100))


@pytest.fixture
def mtf_strategy_data_overlap():
    rng = np.random.default_rng(42)
    return rng.normal(size=(24, 61))


@pytest.fixture
def mtf_strategy_data_no_overlap():
    rng = np.random.default_rng(7)
    return rng.uniform(size=(32, 80)) * 100 - 20


@pytest.fixture
def stft_benchmark_data(pyts_benchmark_data):
    return pyts_benchmark_data


class TestGAFPytsParity:
    @pytest.fixture(autouse=True)
    def _require_pyts(self):
        pytest.importorskip("pyts")

    @pytest.mark.parametrize(
        "pyts_method,industrial_method",
        [
            ("s", "summation"),
            ("d", "difference"),
        ],
    )
    def test_gaf_matches_pyts_benchmark(
        self, pyts_benchmark_data, pyts_method, industrial_method
    ):
        from pyts.image import GramianAngularField

        reference = GramianAngularField(
            method=pyts_method,
            overlapping=GAF_BENCHMARK_PARAMS["overlapping"],
            image_size=GAF_BENCHMARK_PARAMS["image_size"],
        ).fit_transform(pyts_benchmark_data)
        output = GAF(
            {
                "method": industrial_method,
                **GAF_BENCHMARK_PARAMS,
            }
        ).transform(
            torch.as_tensor(pyts_benchmark_data, dtype=torch.float64)
        ).detach().cpu().numpy()

        assert reference.shape == output.shape
        assert np.allclose(reference, output, rtol=1e-5, atol=1e-5)
        rmse = _rmse(reference, output)
        assert rmse < RMSE_TOL_PYTS, f"GAF RMSE {rmse} exceeds {RMSE_TOL_PYTS}"

    def test_gaf_integer_image_size(self, pyts_benchmark_data):
        from pyts.image import GramianAngularField

        data = pyts_benchmark_data[:8]
        reference = GramianAngularField(
            method="s", overlapping=False, image_size=16
        ).fit_transform(data)
        output = GAF(
            {
                "method": "summation",
                "overlapping": False,
                "image_size": 16,
                "torch_device": "cpu",
            }
        ).transform(torch.as_tensor(data, dtype=torch.float64)).detach().cpu().numpy()

        assert reference.shape == output.shape
        assert np.allclose(reference, output, rtol=1e-5, atol=1e-5)
        assert _rmse(reference, output) < RMSE_TOL_PYTS


class TestMTFPytsParity:
    @pytest.fixture(autouse=True)
    def _require_pyts(self):
        pytest.importorskip("pyts")

    def test_mtf_matches_pyts_benchmark(self, pyts_benchmark_data):
        from pyts.image import MarkovTransitionField

        reference = MarkovTransitionField(
            image_size=MTF_BENCHMARK_PARAMS["image_size"],
            n_bins=MTF_BENCHMARK_PARAMS["n_bins"],
            strategy=MTF_BENCHMARK_PARAMS["strategy"],
            overlapping=MTF_BENCHMARK_PARAMS["overlapping"],
            flatten=MTF_BENCHMARK_PARAMS["flatten"],
        ).fit_transform(pyts_benchmark_data)
        output = MTF(MTF_BENCHMARK_PARAMS).transform(
            torch.as_tensor(pyts_benchmark_data, dtype=torch.float64)
        ).detach().cpu().numpy()

        assert reference.shape == output.shape
        rmse = _rmse(reference, output)
        assert rmse < RMSE_TOL_PYTS, f"MTF RMSE {rmse} exceeds {RMSE_TOL_PYTS}"

    @pytest.mark.parametrize("strategy,params", MTF_STRATEGY_CASES)
    def test_mtf_matches_pyts_strategies(
        self,
        strategy,
        params,
        mtf_strategy_data_overlap,
        mtf_strategy_data_no_overlap,
    ):
        from pyts.image import MarkovTransitionField

        data = (
            mtf_strategy_data_overlap
            if params["overlapping"]
            else mtf_strategy_data_no_overlap
        )
        reference = MarkovTransitionField(strategy=strategy, **params).fit_transform(
            data
        )
        output = MTF({**params, "strategy": strategy, "torch_device": "cpu"}).transform(
            torch.as_tensor(data, dtype=torch.float64)
        ).detach().cpu().numpy()

        assert reference.shape == output.shape
        rmse = _rmse(reference, output)
        assert rmse < RMSE_TOL_PYTS, f"MTF RMSE {rmse} exceeds {RMSE_TOL_PYTS}"


class TestSTFTScipyParity:
    @pytest.fixture(autouse=True)
    def _require_scipy(self):
        pytest.importorskip("scipy")

    def test_stft_matches_scipy_benchmark(self, stft_benchmark_data):
        reference = _scipy_stft_reference_batch(
            stft_benchmark_data,
            window_size=STFT_BENCHMARK_PARAMS["window_size"],
            hop_length=STFT_BENCHMARK_PARAMS["hop_length"],
            n_fft=STFT_BENCHMARK_PARAMS["n_fft"],
            power=STFT_BENCHMARK_PARAMS["power"],
        )
        output = STFTSpectrogram(STFT_BENCHMARK_PARAMS).transform(
            torch.as_tensor(stft_benchmark_data, dtype=torch.float64)
        ).detach().cpu().numpy()

        assert reference.shape == output.shape
        assert np.allclose(reference, output, rtol=1e-5, atol=1e-5)
        rmse = _rmse(reference, output)
        assert rmse < RMSE_TOL_SCIPY, f"STFT RMSE {rmse} exceeds {RMSE_TOL_SCIPY}"

    def test_stft_power_one_matches_scipy(self, stft_benchmark_data):
        params = {**STFT_BENCHMARK_PARAMS, "power": 1.0}
        reference = _scipy_stft_reference_batch(
            stft_benchmark_data[:8],
            window_size=params["window_size"],
            hop_length=params["hop_length"],
            n_fft=params["n_fft"],
            power=params["power"],
        )
        output = STFTSpectrogram(params).transform(
            torch.as_tensor(stft_benchmark_data[:8], dtype=torch.float64)
        ).detach().cpu().numpy()

        assert reference.shape == output.shape
        assert np.allclose(reference, output, rtol=1e-5, atol=1e-5)
