from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TrajectoryEmbeddingDiagnostics:
    kind: str
    series_length: int
    window_size: int
    stride: int
    n_windows: int
    n_features: int
    channel_count: int = 1
    original_length: int | None = None


@dataclass(frozen=True)
class TrajectoryEmbeddingResult:
    matrix: np.ndarray
    diagnostics: TrajectoryEmbeddingDiagnostics


@dataclass(frozen=True)
class RankTruncationResult:
    reconstructed_matrix: np.ndarray
    projected_states: np.ndarray
    basis: np.ndarray
    singular_values: np.ndarray
    selected_rank: int
    explained_variance_retained: float


def _normalize_univariate_series(time_series: Sequence[float] | np.ndarray) -> np.ndarray:
    normalized = np.asarray(time_series, dtype=float).reshape(-1)
    if normalized.size < 2:
        raise ValueError('Time series must contain at least two values.')
    return normalized


def normalize_multivariate_series(time_series: Sequence[float] | np.ndarray) -> np.ndarray:
    normalized = np.asarray(time_series, dtype=float)
    if normalized.ndim == 1:
        return normalized.reshape(-1, 1)
    if normalized.ndim != 2:
        raise ValueError('Time series must be 1D or 2D.')
    if normalized.shape[0] < normalized.shape[1] and normalized.shape[0] <= 8:
        normalized = normalized.T
    return normalized


def estimate_window(
        series_length: int,
        forecast_horizon: int | None = None,
        min_ratio: float = 0.10,
        max_ratio: float = 0.35,
) -> int:
    if series_length < 4:
        return max(2, series_length - 1)
    min_window = max(2, int(round(series_length * min_ratio)))
    max_window = max(min_window, int(round(series_length * max_ratio)))
    candidate = int(round((min_window + max_window) / 2))
    if forecast_horizon is not None:
        candidate = max(candidate, int(forecast_horizon) + 1)
    return int(min(max(candidate, min_window), max_window, series_length - 1))


def build_hankel(
        time_series: Sequence[float] | np.ndarray,
        window_size: int,
        stride: int = 1,
) -> TrajectoryEmbeddingResult:
    series = _normalize_univariate_series(time_series)
    resolved_window = int(max(2, min(window_size, len(series) - 1)))
    resolved_stride = int(max(1, stride))
    windows = [
        series[index:index + resolved_window]
        for index in range(0, len(series) - resolved_window + 1, resolved_stride)
    ]
    matrix = np.asarray(windows, dtype=float)
    diagnostics = TrajectoryEmbeddingDiagnostics(
        kind='hankel',
        series_length=int(len(series)),
        window_size=resolved_window,
        stride=resolved_stride,
        n_windows=int(matrix.shape[0]),
        n_features=int(matrix.shape[1]),
    )
    return TrajectoryEmbeddingResult(matrix=matrix, diagnostics=diagnostics)


def build_page(
        time_series: Sequence[float] | np.ndarray,
        window_size: int,
        stride: int | None = None,
) -> TrajectoryEmbeddingResult:
    series = _normalize_univariate_series(time_series)
    resolved_window = int(max(2, min(window_size, len(series))))
    resolved_stride = int(max(1, stride or resolved_window))
    windows = [
        series[index:index + resolved_window]
        for index in range(0, len(series) - resolved_window + 1, resolved_stride)
        if len(series[index:index + resolved_window]) == resolved_window
    ]
    matrix = np.asarray(windows, dtype=float)
    diagnostics = TrajectoryEmbeddingDiagnostics(
        kind='page',
        series_length=int(len(series)),
        window_size=resolved_window,
        stride=resolved_stride,
        n_windows=int(matrix.shape[0]),
        n_features=int(matrix.shape[1]) if matrix.size else resolved_window,
        original_length=int(len(series)),
    )
    return TrajectoryEmbeddingResult(matrix=matrix, diagnostics=diagnostics)


def stack_multivariate(matrices: Sequence[np.ndarray]) -> np.ndarray:
    normalized = [np.asarray(matrix, dtype=float) for matrix in matrices]
    if not normalized:
        raise ValueError('At least one matrix is required for multivariate stacking.')
    reference_shape = normalized[0].shape
    if any(matrix.shape != reference_shape for matrix in normalized):
        raise ValueError('All matrices must have the same shape for multivariate stacking.')
    return np.concatenate(normalized, axis=0)


def split_multivariate(stacked_matrix: np.ndarray, channel_count: int) -> tuple[np.ndarray, ...]:
    matrix = np.asarray(stacked_matrix, dtype=float)
    if channel_count <= 0:
        raise ValueError('channel_count must be positive.')
    if matrix.shape[0] % channel_count != 0:
        raise ValueError('Stacked matrix row count must be divisible by channel_count.')
    rows_per_channel = matrix.shape[0] // channel_count
    return tuple(matrix[index * rows_per_channel:(index + 1) * rows_per_channel] for index in range(channel_count))


def decode_diagonal_average(trajectory_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(trajectory_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError('trajectory_matrix must be 2D.')
    n_windows, window_size = matrix.shape
    reconstructed = np.zeros(n_windows + window_size - 1, dtype=float)
    counts = np.zeros_like(reconstructed)
    for window_index in range(n_windows):
        reconstructed[window_index:window_index + window_size] += matrix[window_index]
        counts[window_index:window_index + window_size] += 1.0
    return reconstructed / np.maximum(counts, 1.0)


def decode_page(page_matrix: np.ndarray, original_length: int | None = None) -> np.ndarray:
    matrix = np.asarray(page_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError('page_matrix must be 2D.')
    reconstructed = matrix.reshape(-1)
    if original_length is not None:
        reconstructed = reconstructed[:original_length]
    return reconstructed


def truncate_rank(
        matrix: np.ndarray,
        rank: int | None = None,
        explained_variance: float = 0.95,
        min_rank: int = 1,
) -> RankTruncationResult:
    normalized = np.asarray(matrix, dtype=float)
    if normalized.ndim != 2:
        raise ValueError('matrix must be 2D.')
    if normalized.size == 0:
        raise ValueError('matrix must not be empty.')

    U, singular_values, VT = np.linalg.svd(normalized, full_matrices=False)
    min_dim = int(min(normalized.shape))
    if rank is None:
        energy = singular_values ** 2
        total_energy = float(np.sum(energy))
        if total_energy <= 0:
            selected_rank = min_dim
        else:
            cumulative = np.cumsum(energy) / total_energy
            selected_rank = int(np.searchsorted(cumulative, explained_variance) + 1)
    else:
        selected_rank = int(rank)

    selected_rank = int(max(min_rank, min(selected_rank, min_dim)))
    projected_states = U[:, :selected_rank] * singular_values[:selected_rank]
    basis = VT[:selected_rank, :].T
    reconstructed = projected_states @ basis.T
    retained = float(np.sum(singular_values[:selected_rank] ** 2) / np.sum(singular_values ** 2)) \
        if np.sum(singular_values ** 2) > 0 else 1.0
    return RankTruncationResult(
        reconstructed_matrix=reconstructed,
        projected_states=projected_states,
        basis=basis,
        singular_values=singular_values,
        selected_rank=selected_rank,
        explained_variance_retained=retained,
    )
