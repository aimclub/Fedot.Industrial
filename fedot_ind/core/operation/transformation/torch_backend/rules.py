"""Shared validation rules for torch-backed transformations."""

from __future__ import annotations

from enum import Enum
from typing import Any


class STFTWindowType(str, Enum):
    hann = "hann"
    hamming = "hamming"
    gaussian = "gaussian"


class GAFMethod(str, Enum):
    gasf = "gasf"
    gadf = "gadf"


class BinningStrategy(str, Enum):
    uniform = "uniform"
    quantile = "quantile"
    normal = "normal"


_GAF_METHOD_ALIASES = {
    "summation": GAFMethod.gasf,
    "s": GAFMethod.gasf,
    "gasf": GAFMethod.gasf,
    "difference": GAFMethod.gadf,
    "d": GAFMethod.gadf,
    "gadf": GAFMethod.gadf,
}


def normalize_binning_strategy(
    value: BinningStrategy | str,
) -> BinningStrategy:
    if isinstance(value, BinningStrategy):
        return value
    try:
        return BinningStrategy(str(value).lower())
    except ValueError as exc:
        known = [item.value for item in BinningStrategy]
        raise ValueError(
            f"Unknown binning strategy {value!r}. Known values: {known}."
        ) from exc


def validate_kbins_params(
    n_bins: int,
    strategy: BinningStrategy | str,
) -> BinningStrategy:
    if n_bins < 2:
        raise ValueError(f"'n_bins' must be >= 2, got {n_bins}.")
    return normalize_binning_strategy(strategy)


def normalize_gaf_method(value: GAFMethod | str) -> GAFMethod:
    if isinstance(value, GAFMethod):
        return value
    try:
        return _GAF_METHOD_ALIASES[str(value).lower()]
    except KeyError as exc:
        raise ValueError(
            "Unknown GAF method. Known values: "
            f"{sorted(_GAF_METHOD_ALIASES)}; got {value!r}."
        ) from exc


def normalize_stft_window_type(
    value: STFTWindowType | str,
) -> STFTWindowType:
    if isinstance(value, STFTWindowType):
        return value
    try:
        return STFTWindowType(str(value).lower())
    except ValueError as exc:
        known = [item.value for item in STFTWindowType]
        raise ValueError(
            f"Unknown STFT window_type {value!r}. Known values: {known}."
        ) from exc


def validate_stft_fft_size(n_fft: int, window_size: int) -> None:
    if n_fft < window_size:
        raise ValueError("'n_fft' must be >= 'window_size'.")


def validate_min_series_length(
    n_timestamps: int,
    *,
    operation: str,
) -> None:
    if n_timestamps < 2:
        raise ValueError(
            f"Time series length must be >= 2 for {operation}, got {n_timestamps}."
        )


def validate_stft_fft_fits_series(
    n_timestamps: int,
    n_fft: int,
    *,
    center: bool,
) -> None:
    if not center and n_timestamps < n_fft:
        raise ValueError(
            f"Time series length ({n_timestamps}) must be >= n_fft "
            f"({n_fft}) when center=False."
        )


def validate_stft_window_fits_series(
    n_timestamps: int,
    window_size: int,
) -> None:
    if n_timestamps < window_size:
        raise ValueError(
            f"Time series length ({n_timestamps}) must be >= window_size "
            f"({window_size})."
        )


def normalize_image_size(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("'image_size' must be int or float.")
    if isinstance(value, int):
        if value < 1:
            raise ValueError("Integer 'image_size' must be >= 1.")
        return value
    if not 0.0 < value <= 1.0:
        raise ValueError("Float 'image_size' must be > 0 and <= 1.")
    return value


def validate_image_size_fits_series(
    image_size: int,
    n_timestamps: int,
) -> None:
    if image_size > n_timestamps:
        raise ValueError(
            "Integer 'image_size' must be <= n_timestamps. "
            f"Got image_size={image_size}, n_timestamps={n_timestamps}."
        )


def normalize_optional_window_size(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("'window_size' must be an integer.")
    if value < 1:
        raise ValueError("'window_size' must be >= 1.")
    return value


def validate_window_size(
    window_size: int,
    ts_size: int,
) -> None:
    if window_size < 1:
        raise ValueError("'window_size' must be >= 1.")
    if ts_size < 2:
        raise ValueError("'ts_size' must be >= 2.")
    if window_size > ts_size:
        raise ValueError("'window_size' must be <= n_timestamps.")


def validate_n_segments_min(n_segments: int) -> None:
    if n_segments < 2:
        raise ValueError("'n_segments' must be >= 2.")


def validate_n_segments_fits_series(
    n_segments: int,
    ts_size: int,
) -> None:
    if n_segments > ts_size:
        raise ValueError("'n_segments' must be <= ts_size.")


def validate_mtf_output_layout(
    *,
    flatten: bool,
    return_init_dim: bool,
) -> None:
    if flatten and return_init_dim:
        raise ValueError("'flatten' and 'return_init_dim' cannot both be True.")


def validate_gaf_input_range(
    minimum: float,
    maximum: float,
    *,
    epsilon: float = 1e-5,
) -> None:
    if minimum < -1.0 - epsilon or maximum > 1.0 + epsilon:
        raise ValueError(
            "If 'use_per_sample_minmax' is False, all the values "
            "of X must be between -1 and 1."
        )


def validate_series_input_rank(
    ndim: int,
    shape: tuple[int, ...],
) -> None:
    if ndim < 1 or ndim > 3:
        raise ValueError(f"X must be 1D, 2D or 3D, got shape={shape}")


def validate_flat_batch_size(
    actual: int,
    *,
    batch: int,
    n_channels: int,
    init_shape: tuple[int, ...],
) -> None:
    expected = batch * n_channels
    if actual != expected:
        raise ValueError(
            f"Batch/channel flatten mismatch: input shape {init_shape} "
            f"implies {expected} flat samples, got {actual}."
        )
