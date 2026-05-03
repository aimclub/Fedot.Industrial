import pytest
import torch
import numpy as np
from pyts.image import MarkovTransitionField
from fedot_ind.core.operation.transformation.torch_backend.image_transformation.methods.mtf_transformation import (
    MTF,
)


@pytest.fixture
def data() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(size=(24, 61))


@pytest.fixture
def noisy_rand_data() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.random(size=(32, 80)) * 100.0 - 20.0


@pytest.mark.parametrize("strategy", ["uniform", "quantile", "normal"])
def test_mtf_transform_close_to_pyts(data: np.ndarray, strategy: str) -> None:
    image_size = 0.7
    n_bins = 8
    overlapping = True

    mtf_pyts = MarkovTransitionField(
        image_size=image_size,
        n_bins=n_bins,
        strategy=strategy,
        overlapping=overlapping,
        flatten=False,
    )
    result_pyts = mtf_pyts.fit_transform(data)

    mtf_torch = MTF(
        {
            "image_size": image_size,
            "n_bins": n_bins,
            "strategy": strategy,
            "overlapping": overlapping,
            "flatten": False,
        }
    )
    result_torch = mtf_torch.transform(torch.tensor(data, dtype=torch.float64))
    result_torch_np = result_torch.detach().cpu().numpy()

    assert result_pyts.shape == result_torch_np.shape
    assert np.allclose(result_pyts, result_torch_np, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("strategy", ["uniform", "quantile", "normal"])
def test_mtf_transform_close_to_pyts_noisy_rand(
    noisy_rand_data: np.ndarray, strategy: str
) -> None:
    image_size = 0.55
    n_bins = 8
    overlapping = False

    mtf_pyts = MarkovTransitionField(
        image_size=image_size,
        n_bins=n_bins,
        strategy=strategy,
        overlapping=overlapping,
        flatten=False,
    )
    result_pyts = mtf_pyts.fit_transform(noisy_rand_data)

    mtf_torch = MTF(
        {
            "image_size": image_size,
            "n_bins": n_bins,
            "strategy": strategy,
            "overlapping": overlapping,
            "flatten": False,
        }
    )
    result_torch = mtf_torch.transform(
        torch.tensor(noisy_rand_data, dtype=torch.float64)
    )
    result_torch_np = result_torch.detach().cpu().numpy()

    assert result_pyts.shape == result_torch_np.shape
    assert np.allclose(result_pyts, result_torch_np, atol=1e-8, rtol=1e-6)
