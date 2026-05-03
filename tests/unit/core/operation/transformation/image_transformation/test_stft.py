import numpy as np
import pytest
import torch

from fedot_ind.core.operation.transformation.torch_backend.image_transformation.methods.stft_transformation import (
    STFTSpectrogram,
)


@pytest.fixture
def data() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(size=(24, 61))


@pytest.fixture
def noisy_rand_data() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.random(size=(32, 80)) * 100.0 - 20.0


def _reference_spectrogram_torch_stft(
    X: torch.Tensor, model: STFTSpectrogram
) -> torch.Tensor:
    """Same pipeline as ``STFTSpectrogram.transform`` but spelled out (regression oracle)."""
    window = model._window(X.device, X.dtype)
    stft = torch.stft(
        X,
        n_fft=model.n_fft,
        hop_length=model.hop_length,
        win_length=model.window_size,
        window=window,
        center=model.center,
        pad_mode=model.pad_mode,
        normalized=model.normalized,
        onesided=True,
        return_complex=True,
    )
    return torch.abs(stft).pow(model.power)


@pytest.mark.parametrize("window_type", ["hann", "hamming", "gaussian"])
def test_stft_transform_matches_torch_stft(data: np.ndarray, window_type: str) -> None:
    window_size = 16
    hop_length = 4
    n_fft = 32
    power = 2.0
    sigma = 3.0 if window_type == "gaussian" else None

    params = {
        "window_size": window_size,
        "hop_length": hop_length,
        "n_fft": n_fft,
        "window_type": window_type,
        "center": False,
        "pad_mode": "reflect",
        "power": power,
        "normalized": False,
    }
    if sigma is not None:
        params["sigma"] = sigma

    model = STFTSpectrogram(params)
    X = torch.tensor(data, dtype=torch.float64)
    expected = _reference_spectrogram_torch_stft(X, model)
    actual = model.transform(X)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("window_type", ["hann", "hamming", "gaussian"])
def test_stft_transform_matches_torch_stft_noisy_rand(
    noisy_rand_data: np.ndarray, window_type: str
) -> None:
    window_size = 32
    hop_length = 8
    n_fft = 64
    power = 2.0
    sigma = 5.0 if window_type == "gaussian" else None

    params = {
        "window_size": window_size,
        "hop_length": hop_length,
        "n_fft": n_fft,
        "window_type": window_type,
        "center": False,
        "power": power,
        "normalized": False,
    }
    if sigma is not None:
        params["sigma"] = sigma

    model = STFTSpectrogram(params)
    X = torch.tensor(noisy_rand_data, dtype=torch.float64)
    expected = _reference_spectrogram_torch_stft(X, model)
    actual = model.transform(X)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("center", [False, True])
def test_stft_transform_matches_torch_stft_center(center: bool) -> None:
    params = {
        "window_size": 16,
        "hop_length": 4,
        "n_fft": 32,
        "window_type": "hann",
        "center": center,
        "pad_mode": "reflect",
        "power": 2.0,
        "normalized": False,
    }
    model = STFTSpectrogram(params)
    X = torch.randn(4, 100, dtype=torch.float64)
    expected = _reference_spectrogram_torch_stft(X, model)
    actual = model.transform(X)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_stft_transform_matches_torch_stft_normalized() -> None:
    params = {
        "window_size": 16,
        "hop_length": 4,
        "n_fft": 16,
        "window_type": "hamming",
        "center": False,
        "power": 1.5,
        "normalized": True,
    }
    model = STFTSpectrogram(params)
    X = torch.randn(8, 90, dtype=torch.float64)
    torch.testing.assert_close(
        model.transform(X),
        _reference_spectrogram_torch_stft(X, model),
        rtol=0.0,
        atol=0.0,
    )


def test_stft_batched_matches_stacked_rows(data: np.ndarray) -> None:
    params = {
        "window_size": 16,
        "hop_length": 4,
        "n_fft": 32,
        "window_type": "hann",
        "center": False,
        "power": 2.0,
        "normalized": False,
    }
    m = STFTSpectrogram(params)
    X = torch.tensor(data, dtype=torch.float64)
    batched = m.transform(X)
    stacked = torch.stack([m.transform(X[i : i + 1]).squeeze(0) for i in range(X.shape[0])])
    assert torch.allclose(batched, stacked, atol=0.0, rtol=0.0)


def test_stft_raises_on_unknown_window_type() -> None:
    with pytest.raises(ValueError, match="window_type"):
        STFTSpectrogram({"window_type": "blackman"})


def test_stft_raises_on_non_2d_input() -> None:
    m = STFTSpectrogram(
        {
            "window_size": 16,
            "hop_length": 4,
            "n_fft": 32,
            "center": False,
        }
    )
    with pytest.raises(ValueError, match="2D"):
        m.transform(torch.randn(61))
