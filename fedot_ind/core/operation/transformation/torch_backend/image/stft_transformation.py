from typing import Any, Optional

import torch

from fedot_ind.core.operation.transformation.torch_backend.image.tools import (
    prepare_series_input,
    convert_to_init_dim,
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


def _parse_positive_int(value: Any, name: str, *, default: int) -> int:
    if value is None:
        parsed = default
    else:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"'{name}' must be a positive integer, got {value!r}."
            ) from exc
    if parsed < 1:
        raise ValueError(f"'{name}' must be >= 1, got {parsed}.")
    return parsed


def _parse_positive_float(value: Any, name: str, *, default: float) -> float:
    if value is None:
        parsed = default
    else:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"'{name}' must be a positive number, got {value!r}."
            ) from exc
    if parsed <= 0:
        raise ValueError(f"'{name}' must be > 0, got {parsed}.")
    return parsed


class STFTSpectrogram:
    """
    PyTorch Short-Time Fourier Transform (STFT) spectrogram for time series.

    For each row of the input batch, computes STFT via :func:`torch.stft` and
    returns ``abs(STFT) ** power`` (default ``power=2``), i.e. a power
    spectrogram. Output layout is ``(batch, n_freq_bins, n_frames)``, suitable
    as a 2D "image" per sample (frequency × time).

    Config parameters (``params`` dict):
        window_size (int, default ``256``): Samples per STFT frame
            (``win_length`` in PyTorch).
        hop_length (int, default ``64``): Hop between consecutive frames.
        n_fft (int, default ``window_size``): FFT size; must be
            ``>= window_size``.
        window_type (str, default ``'hann'``): One of ``'hann'``, ``'hamming'``,
            ``'gaussian'``.
        center (bool, default ``True``): Pad the series so frame centers align.
        power (float, default ``2.0``): Exponent applied to STFT magnitude.
        pad_mode (str, default ``'reflect'``): Padding mode when ``center=True``.
        normalized (bool, default ``False``): Window energy normalization for
            :func:`torch.stft`.
        sigma (float or None, default ``None``): Gaussian window std in samples;
            defaults to ``window_size / 6`` when ``window_type='gaussian'``.
        sampling_rate (float or None, default ``None``): Metadata only (Hz);
            does not affect the computation.
        return_init_dim (bool, default ``True``): If ``True``, restore batch/
            channel axes for 3D input ``(B, C, T)`` → ``(B, C, n_freq, n_frames)``.
            For 1D/2D inputs the output batch layout is left unchanged.
        torch_device (str, default ``'auto'``): Device to use for the transformation.
    """

    def __init__(self, params: Optional[dict[str, Any]] = None):
        params = params or {}
        self.window_size = _parse_positive_int(
            params.get("window_size", 256), "window_size", default=256
        )
        self.hop_length = _parse_positive_int(
            params.get("hop_length", 64), "hop_length", default=64
        )
        self.window_type = str(params.get("window_type", "hann")).lower()
        self.n_fft = _parse_positive_int(
            params.get("n_fft", self.window_size), "n_fft", default=self.window_size
        )
        self.center = bool(params.get("center", True))
        self.pad_mode = str(params.get("pad_mode", "reflect"))
        self.power = _parse_positive_float(params.get("power", 2.0), "power", default=2.0)
        self.normalized = bool(params.get("normalized", False))
        self.sampling_rate = params.get("sampling_rate", None)
        self.sigma = params.get("sigma", None)
        if self.sigma is not None:
            self.sigma = _parse_positive_float(self.sigma, "sigma", default=1.0)
        self.return_init_dim = bool(params.get("return_init_dim", True))
        self.torch_device = params.get("torch_device", "auto")
        self._check_params()

    def _window(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        w = self.window_size
        if self.window_type == "hann":
            return torch.hann_window(w, periodic=True, device=device, dtype=dtype)
        if self.window_type == "hamming":
            return torch.hamming_window(w, periodic=True, device=device, dtype=dtype)
        sigma = self.sigma if self.sigma is not None else self.window_size / 6.0
        return _gaussian_window(w, sigma, device=device, dtype=dtype)

    def _check_params(self) -> None:
        if self.n_fft < self.window_size:
            raise ValueError("'n_fft' must be >= 'window_size'.")
        if self.window_type not in ("hann", "hamming", "gaussian"):
            raise ValueError(
                "'window_type' must be one of: hann, hamming, gaussian "
                f"(got {self.window_type!r})."
            )
        if self.window_type == "gaussian":
            sig = self.sigma if self.sigma is not None else self.window_size / 6.0
            if sig <= 0:
                raise ValueError("'sigma' must be > 0 for gaussian window.")

    def _check_series_length(self, n_timestamps: int) -> None:
        if n_timestamps < 2:
            raise ValueError(
                f"Time series length must be >= 2 for STFT, got {n_timestamps}."
            )
        if not self.center and n_timestamps < self.n_fft:
            raise ValueError(
                f"Time series length ({n_timestamps}) must be >= n_fft "
                f"({self.n_fft}) when center=False."
            )
        if n_timestamps < self.window_size:
            raise ValueError(
                f"Time series length ({n_timestamps}) must be >= window_size "
                f"({self.window_size})."
            )

    def transform(self, X: Any) -> torch.Tensor:
        """
        Compute the STFT spectrogram for a batch of 1D time series.

        Args:
            X: Tensor of shape ``(batch, n_timestamps)`` (real-valued).

        Returns:
            Tensor of shape ``(batch, n_freq_bins, n_frames)`` with
            ``n_freq_bins = n_fft // 2 + 1`` (one-sided STFT for real input).
        """

        X, init_shape = prepare_series_input(X, torch_device=self.torch_device)
        self._check_series_length(X.shape[1])

        device = X.device
        dtype = X.dtype
        window = self._window(device, dtype)

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

        if self.return_init_dim:
            spec = convert_to_init_dim(spec, init_shape)
        return spec
