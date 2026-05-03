from typing import Any, Optional

import torch

from fedot.core.operations.operation_parameters import OperationParameters
from fedot_ind.core.operation.transformation.torch_backend.image_transformation.methods.gaf_transformation import (
    register_transformer,
)


def _gaussian_window(
    length: int,
    sigma: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Symmetric Gaussian window of shape (length,) (torch-only, no scipy)."""
    n = torch.arange(length, device=device, dtype=dtype)
    center = (length - 1) / 2.0
    return torch.exp(-0.5 * ((n - center) / sigma) ** 2)


@register_transformer("stft")
class STFTSpectrogram:
    """
    PyTorch Short-Time Fourier Transform (STFT) spectrogram for time series.

    For each row of the input batch, computes STFT via :func:`torch.stft` and
    returns ``abs(STFT) ** power`` (default ``power=2``), i.e. a power
    spectrogram. Output layout is ``(batch, n_freq_bins, n_frames)``, suitable
    as a 2D "image" per sample (frequency × time).

    Supported windows: Hann, Hamming, Gaussian (Gaussian uses ``sigma`` in
    samples). ``sampling_rate`` is stored as metadata only and does not affect
    the computation.

    Attributes:
        window_size: Samples per STFT frame (``win_length`` in PyTorch).
        hop_length: Hop between consecutive frames.
        window_type: One of ``"hann"``, ``"hamming"``, ``"gaussian"``.
        n_fft: FFT size (must be >= ``window_size``).
        center: If True, pad the series so frame centers align (see ``pad_mode``).
        pad_mode: Padding mode when ``center`` is True (e.g. ``"reflect"``).
        power: Exponent applied to the magnitude (2.0 → power spectrogram).
        normalized: Passed to :func:`torch.stft` (window energy normalization).
        sampling_rate: Optional metadata (Hz); not used in the transform.
        sigma: Standard deviation in samples for Gaussian window; if None and
            ``window_type`` is ``"gaussian"``, defaults to ``window_size / 6``.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        params = params or {}
        self.window_size = int(params.get("window_size", 256))
        self.hop_length = int(params.get("hop_length", 64))
        self.window_type = str(params.get("window_type", "hann")).lower()
        self.n_fft = int(params.get("n_fft", self.window_size))
        self.center = bool(params.get("center", True))
        self.pad_mode = str(params.get("pad_mode", "reflect"))
        self.power = float(params.get("power", 2.0))
        self.normalized = bool(params.get("normalized", False))
        self.sampling_rate = params.get("sampling_rate", None)
        self.sigma = params.get("sigma", None)
        if self.sigma is not None:
            self.sigma = float(self.sigma)

        if self.window_size < 1:
            raise ValueError("'window_size' must be >= 1.")
        if self.hop_length < 1:
            raise ValueError("'hop_length' must be >= 1.")
        if self.n_fft < self.window_size:
            raise ValueError("'n_fft' must be >= 'window_size'.")
        if self.power <= 0:
            raise ValueError("'power' must be > 0.")
        if self.window_type not in ("hann", "hamming", "gaussian"):
            raise ValueError(
                "'window_type' must be one of: hann, hamming, gaussian "
                f"(got {self.window_type!r})."
            )
        if self.window_type == "gaussian":
            sig = self.sigma if self.sigma is not None else self.window_size / 6.0
            if sig <= 0:
                raise ValueError("'sigma' must be > 0 for gaussian window.")

    def _window(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        w = self.window_size
        if self.window_type == "hann":
            # periodic=True matches common STFT / librosa-style tapering
            return torch.hann_window(w, periodic=True, device=device, dtype=dtype)
        if self.window_type == "hamming":
            return torch.hamming_window(w, periodic=True, device=device, dtype=dtype)
        sigma = self.sigma if self.sigma is not None else self.window_size / 6.0
        return _gaussian_window(w, sigma, device=device, dtype=dtype)

    def transform(self, X: Any) -> torch.Tensor:
        """
        Compute the STFT spectrogram for a batch of 1D time series.

        Args:
            X: Tensor of shape ``(batch, n_timestamps)`` (real-valued).

        Returns:
            Tensor of shape ``(batch, n_freq_bins, n_frames)`` with
            ``n_freq_bins = n_fft // 2 + 1`` (one-sided STFT for real input).
        """
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (batch, time), got shape={tuple(X.shape)}")
        if not torch.is_floating_point(X):
            X = X.float()

        device = X.device
        dtype = X.dtype
        window = self._window(device, dtype)

        # torch.stft: last dimension is time; batch dimensions are preserved
        stft = torch.stft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_size,
            window=window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
            onesided=True,
            return_complex=True,
        )
        spec = torch.abs(stft).pow(self.power)
        return spec
