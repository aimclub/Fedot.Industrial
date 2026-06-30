"""Unit tests for torch image transformations (GAF, MTF, STFT)."""

from __future__ import annotations
from fedot_ind.core.operation.transformation.torch_backend.image.tools import (
    check_input_shape,
    convert_to_init_dim,
    prepare_series_input,
)
from fedot_ind.core.operation.transformation.torch_backend.image.stft_transformation import (
    STFTSpectrogram,
)
from fedot_ind.core.operation.transformation.torch_backend.image.mtf_transformation import MTF
from fedot_ind.core.operation.transformation.torch_backend.image.gaf_transformation import GAF
from fedot_ind.core.kernel_learning import resolve_torch_device

import numpy as np
import pytest
import torch

pytest.importorskip("torch")


T = 64
B = 3
C = 2
IMAGE_SIDE = 16

STFT_PARAMS = {
    "window_size": 16,
    "hop_length": 4,
    "n_fft": 16,
    "torch_device": "cpu",
}


def assert_finite(tensor: torch.Tensor) -> None:
    assert torch.isfinite(tensor).all(), "Output contains NaN or inf"


def expected_stft_freq_frames(
    n_timestamps: int,
    *,
    n_fft: int,
    hop_length: int,
    window_size: int,
    center: bool = True,
) -> tuple[int, int]:
    x = torch.randn(1, n_timestamps)
    window = torch.hann_window(window_size, periodic=True)
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=window_size,
        window=window,
        center=center,
        return_complex=True,
    )
    return spec.shape[1], spec.shape[2]


class TestInputUtilities:
    @pytest.mark.parametrize(
        "shape,expected_working",
        [
            ((T,), (1, T)),
            ((B, T), (B, T)),
            ((B, C, T), (B * C, T)),
        ],
    )
    def test_check_input_shape_supported_ndim(self, shape, expected_working):
        x = torch.randn(*shape)
        out, init_shape = check_input_shape(x)
        assert out.shape == expected_working
        assert init_shape == shape

    def test_check_input_shape_rejects_4d(self):
        with pytest.raises(ValueError, match="1D, 2D or 3D"):
            check_input_shape(torch.randn(2, 3, 4, 5))

    def test_prepare_series_input_respects_torch_device_cpu(self):
        x = np.random.randn(B, T).astype(np.float32)
        out, init_shape = prepare_series_input(x, torch_device="cpu")
        assert str(out.device) == "cpu"
        assert out.dtype == torch.float32
        assert init_shape == (B, T)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prepare_series_input_respects_torch_device_cuda(self):
        x = torch.randn(B, T)
        out, _ = prepare_series_input(x, torch_device="cuda")
        assert out.device.type == "cuda"

    def test_convert_to_init_dim_restores_multichannel(self):
        flat = torch.randn(B * C, IMAGE_SIDE, IMAGE_SIDE)
        restored = convert_to_init_dim(flat, (B, C, T))
        assert restored.shape == (B, C, IMAGE_SIDE, IMAGE_SIDE)

    def test_convert_to_init_dim_passthrough_batch(self):
        batch = torch.randn(B, IMAGE_SIDE, IMAGE_SIDE)
        out = convert_to_init_dim(batch, (B, T))
        assert out.shape == (B, IMAGE_SIDE, IMAGE_SIDE)

    def test_convert_to_init_dim_passthrough_single_series(self):
        single = torch.randn(1, IMAGE_SIDE, IMAGE_SIDE)
        out = convert_to_init_dim(single, (T,))
        assert out.shape == (1, IMAGE_SIDE, IMAGE_SIDE)


class TestGAF:
    @pytest.fixture
    def transformer(self):
        return GAF({"image_size": IMAGE_SIDE, "torch_device": "cpu"})

    @pytest.mark.parametrize(
        "shape,expected",
        [
            ((T,), (1, IMAGE_SIDE, IMAGE_SIDE)),
            ((B, T), (B, IMAGE_SIDE, IMAGE_SIDE)),
            ((B, C, T), (B, C, IMAGE_SIDE, IMAGE_SIDE)),
        ],
    )
    def test_output_shapes(self, transformer, shape, expected):
        out = transformer.transform(torch.randn(*shape))
        assert out.shape == expected

    def test_cnn_multichannel_layout(self, transformer):
        out = transformer.transform(torch.randn(B, C, T))
        assert out.shape == (B, C, IMAGE_SIDE, IMAGE_SIDE)

    def test_return_init_dim_false_flattens_channels_to_batch(self):
        gaf = GAF(
            {"image_size": IMAGE_SIDE, "return_init_dim": False, "torch_device": "cpu"}
        )
        out = gaf.transform(torch.randn(B, C, T))
        assert out.shape == (B * C, IMAGE_SIDE, IMAGE_SIDE)

    @pytest.mark.parametrize("method", ["summation", "s", "gasf", "difference", "d", "gadf"])
    def test_method_aliases(self, method):
        gaf = GAF({"image_size": IMAGE_SIDE, "method": method, "torch_device": "cpu"})
        out = gaf.transform(torch.randn(B, T))
        assert out.shape == (B, IMAGE_SIDE, IMAGE_SIDE)
        assert_finite(out)

    def test_overlapping_when_not_divisible(self):
        gaf = GAF(
            {
                "image_size": 0.5,
                "overlapping": True,
                "torch_device": "cpu",
            }
        )
        out = gaf.transform(torch.randn(B, 65))
        assert out.shape == (B, 33, 33)
        assert_finite(out)

    def test_image_size_float_fraction(self):
        gaf = GAF({"image_size": 0.5, "torch_device": "cpu"})
        out = gaf.transform(torch.randn(B, T))
        assert out.shape == (B, T // 2, T // 2)

    def test_config_not_mutated_on_repeated_transform(self):
        gaf = GAF({"image_size": 0.5, "torch_device": "cpu"})
        out_short = gaf.transform(torch.randn(2, 100))
        out_long = gaf.transform(torch.randn(2, 200))
        assert gaf.image_size == 0.5
        assert out_short.shape[-2:] == (50, 50)
        assert out_long.shape[-2:] == (100, 100)

    @pytest.mark.parametrize(
        "params,match",
        [
            ({"method": "invalid"}, "method"),
            ({"image_size": 0.0}, "image_size"),
            ({"image_size": 100, "torch_device": "cpu"}, "<= n_timestamps"),
        ],
    )
    def test_invalid_params_raise(self, params, match):
        gaf = GAF({**params, "torch_device": "cpu"})
        with pytest.raises((ValueError, TypeError), match=match):
            gaf.transform(torch.randn(B, T))

    def test_short_series_raises(self):
        gaf = GAF({"image_size": IMAGE_SIDE, "torch_device": "cpu"})
        with pytest.raises(ValueError, match=">= 2"):
            gaf.transform(torch.randn(1))

    def test_finite_on_random_and_constant_series(self, transformer):
        assert_finite(transformer.transform(torch.randn(B, C, T)))
        assert_finite(transformer.transform(torch.full((B, C, T), 3.14)))

    def test_torch_device_matches_resolve_torch_device(self, transformer):
        out = transformer.transform(torch.randn(B, T))
        expected = str(resolve_torch_device("cpu"))
        assert str(out.device) == expected


class TestMTF:
    @pytest.fixture
    def transformer(self):
        return MTF({"image_size": IMAGE_SIDE, "torch_device": "cpu"})

    @pytest.mark.parametrize(
        "shape,expected",
        [
            ((T,), (1, IMAGE_SIDE, IMAGE_SIDE)),
            ((B, T), (B, IMAGE_SIDE, IMAGE_SIDE)),
            ((B, C, T), (B, C, IMAGE_SIDE, IMAGE_SIDE)),
        ],
    )
    def test_output_shapes(self, transformer, shape, expected):
        out = transformer.transform(torch.randn(*shape))
        assert out.shape == expected

    @pytest.mark.parametrize("strategy", ["uniform", "quantile", "normal"])
    def test_strategy_config(self, strategy):
        mtf = MTF({"image_size": IMAGE_SIDE, "strategy": strategy, "torch_device": "cpu"})
        out = mtf.transform(torch.randn(B, T))
        assert out.shape == (B, IMAGE_SIDE, IMAGE_SIDE)
        assert_finite(out)

    def test_overlapping_runs(self):
        mtf = MTF(
            {
                "image_size": 0.5,
                "overlapping": True,
                "torch_device": "cpu",
            }
        )
        out = mtf.transform(torch.randn(B, 65))
        assert out.shape[0] == B
        assert_finite(out)

    def test_flatten_output(self):
        mtf = MTF(
            {
                "image_size": IMAGE_SIDE,
                "flatten": True,
                "return_init_dim": False,
                "torch_device": "cpu",
            }
        )
        out = mtf.transform(torch.randn(B, T))
        assert out.shape == (B, IMAGE_SIDE * IMAGE_SIDE)

    def test_flatten_and_return_init_dim_conflict(self):
        with pytest.raises(ValueError, match="flatten.*return_init_dim"):
            MTF({"flatten": True, "return_init_dim": True})

    @pytest.mark.parametrize(
        "params,match",
        [
            ({"n_bins": 1}, "n_bins"),
            ({"strategy": "bad"}, "strategy"),
        ],
    )
    def test_invalid_init_params(self, params, match):
        with pytest.raises(ValueError, match=match):
            MTF({**params, "torch_device": "cpu"})

    def test_invalid_image_size_at_transform(self):
        mtf = MTF({"image_size": 0.0, "torch_device": "cpu"})
        with pytest.raises(ValueError, match="image_size"):
            mtf.transform(torch.randn(B, T))
        mtf_int = MTF({"image_size": 100, "torch_device": "cpu"})
        with pytest.raises(ValueError, match="n_timestamps"):
            mtf_int.transform(torch.randn(B, T))

    def test_short_series_raises(self, transformer):
        with pytest.raises(ValueError, match=">= 2"):
            transformer.transform(torch.randn(1))

    def test_finite_on_random_and_constant_series(self, transformer):
        assert_finite(transformer.transform(torch.randn(B, C, T)))
        assert_finite(transformer.transform(torch.zeros(B, C, T)))


class TestSTFTSpectrogram:
    @pytest.fixture
    def transformer(self):
        return STFTSpectrogram(STFT_PARAMS)

    @pytest.mark.parametrize(
        "shape,expected_batch",
        [
            ((T,), 1),
            ((B, T), B),
            ((B, C, T), B),
        ],
    )
    def test_output_shapes(self, transformer, shape, expected_batch):
        n_freq, n_frames = expected_stft_freq_frames(
            shape[-1],
            n_fft=STFT_PARAMS["n_fft"],
            hop_length=STFT_PARAMS["hop_length"],
            window_size=STFT_PARAMS["window_size"],
            center=True,
        )
        out = transformer.transform(torch.randn(*shape))
        if len(shape) == 3:
            assert out.shape == (expected_batch, shape[1], n_freq, n_frames)
        else:
            assert out.shape == (expected_batch, n_freq, n_frames)

    def test_cnn_multichannel_layout(self, transformer):
        n_freq, n_frames = expected_stft_freq_frames(
            T,
            n_fft=STFT_PARAMS["n_fft"],
            hop_length=STFT_PARAMS["hop_length"],
            window_size=STFT_PARAMS["window_size"],
        )
        out = transformer.transform(torch.randn(B, C, T))
        assert out.shape == (B, C, n_freq, n_frames)

    @pytest.mark.parametrize("window_type", ["hann", "hamming", "gaussian"])
    def test_window_type_config(self, window_type):
        stft = STFTSpectrogram({**STFT_PARAMS, "window_type": window_type})
        out = stft.transform(torch.randn(B, T))
        assert out.ndim == 3
        assert_finite(out)

    def test_center_and_power_config(self):
        stft_p1 = STFTSpectrogram({**STFT_PARAMS, "center": False, "power": 1.0})
        stft_p2 = STFTSpectrogram({**STFT_PARAMS, "center": False, "power": 2.0})
        x = torch.randn(1, T)
        out_p1 = stft_p1.transform(x)
        out_p2 = stft_p2.transform(x)
        assert_finite(out_p1)
        assert_finite(out_p2)
        assert torch.allclose(out_p2, out_p1**2, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize(
        "params,match",
        [
            ({"n_fft": 8, "window_size": 16}, "n_fft"),
            ({"window_type": "boxcar"}, "window_type"),
            ({"power": 0}, "power"),
            ({"window_size": "bad"}, "window_size"),
        ],
    )
    def test_invalid_init_params(self, params, match):
        merged = {**STFT_PARAMS, **params}
        with pytest.raises(ValueError, match=match):
            STFTSpectrogram(merged)

    def test_short_series_raises(self, transformer):
        with pytest.raises(ValueError, match=">= 2"):
            transformer.transform(torch.randn(1))

    def test_series_shorter_than_window_raises(self):
        stft = STFTSpectrogram(
            {**STFT_PARAMS, "window_size": 32, "n_fft": 32}
        )
        with pytest.raises(ValueError, match="window_size"):
            stft.transform(torch.randn(B, 20))

    def test_center_false_requires_n_fft_length(self):
        stft = STFTSpectrogram(
            {**STFT_PARAMS, "center": False, "n_fft": 32, "window_size": 16}
        )
        with pytest.raises(ValueError, match="n_fft"):
            stft.transform(torch.randn(B, 20))

    def test_finite_on_random_and_constant_series(self, transformer):
        assert_finite(transformer.transform(torch.randn(B, C, T)))
        assert_finite(transformer.transform(torch.ones(B, C, T)))

    def test_torch_device_cpu(self, transformer):
        out = transformer.transform(np.random.randn(B, T).astype(np.float32))
        assert str(out.device) == str(resolve_torch_device("cpu"))


class TestSmallBatchFiniteValues:
    """Smoke tests on tiny synthetic batches for all transformations."""

    @pytest.mark.parametrize(
        "cls,params",
        [
            (GAF, {"image_size": IMAGE_SIDE, "torch_device": "cpu"}),
            (MTF, {"image_size": IMAGE_SIDE, "torch_device": "cpu"}),
            (STFTSpectrogram, STFT_PARAMS),
        ],
    )
    def test_multichannel_batch_finite(self, cls, params):
        transformer = cls(params)
        out = transformer.transform(torch.randn(2, 3, T))
        assert_finite(out)
        assert out.shape[0] == 2
        if isinstance(transformer, STFTSpectrogram):
            assert out.shape[1] == 3
        else:
            assert out.shape[1] == 3
            assert out.shape[2] == out.shape[3]
