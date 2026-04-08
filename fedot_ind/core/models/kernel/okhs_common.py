from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Sequence

import numpy as np

from fedot_ind.core.operation.transformation.data.trajectory_embedding import (
    build_hankel,
    truncate_rank,
)


class OKHSMethod(str, Enum):
    DMD = "dmd"
    DIRECT = "direct"
    OCCUPATION = "occupation"


class QPolicy(str, Enum):
    FIXED = "fixed"
    DATA_DRIVEN = "data_driven"


class WindowPolicy(str, Enum):
    FIXED = "fixed"
    ADAPTIVE_HEURISTIC = "adaptive_heuristic"
    ADAPTIVE_CYCLE_AWARE = "adaptive_cycle_aware"


class TrajectorySamplingPolicy(str, Enum):
    DENSE = "dense"
    ADAPTIVE_STRIDE = "adaptive_stride"


class TrajectoryRankPolicy(str, Enum):
    NONE = "none"
    EXPLAINED_DISPERSION = "explained_dispersion"


class TrajectoryRepresentationPolicy(str, Enum):
    NONE = "none"
    RECONSTRUCTED = "reconstructed"
    PROJECTED = "projected"


class LatentTrajectoryStridePolicy(str, Enum):
    DENSE = "dense"
    ADAPTIVE = "adaptive"


def normalize_okhs_method(method: str | OKHSMethod) -> OKHSMethod:
    if isinstance(method, OKHSMethod):
        return method

    normalized = str(method).strip().lower()
    aliases = {
        "dmd": OKHSMethod.DMD,
        "direct": OKHSMethod.DIRECT,
        "occupation": OKHSMethod.OCCUPATION,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported OKHS method: {method}")
    return aliases[normalized]


def normalize_q_policy(q_policy: str | QPolicy) -> QPolicy:
    if isinstance(q_policy, QPolicy):
        return q_policy

    normalized = str(q_policy).strip().lower()
    aliases = {
        "fixed": QPolicy.FIXED,
        "data_driven": QPolicy.DATA_DRIVEN,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported q_policy: {q_policy}")
    return aliases[normalized]


def normalize_window_policy(window_policy: str | WindowPolicy) -> WindowPolicy:
    if isinstance(window_policy, WindowPolicy):
        return window_policy

    normalized = str(window_policy).strip().lower()
    aliases = {
        "fixed": WindowPolicy.FIXED,
        "adaptive": WindowPolicy.ADAPTIVE_HEURISTIC,
        "adaptive_heuristic": WindowPolicy.ADAPTIVE_HEURISTIC,
        "adaptive_cycle_aware": WindowPolicy.ADAPTIVE_CYCLE_AWARE,
        "cycle_aware": WindowPolicy.ADAPTIVE_CYCLE_AWARE,
        "seasonal": WindowPolicy.ADAPTIVE_CYCLE_AWARE,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported window_policy: {window_policy}")
    return aliases[normalized]


def normalize_trajectory_sampling_policy(
        trajectory_sampling_policy: str | TrajectorySamplingPolicy,
) -> TrajectorySamplingPolicy:
    if isinstance(trajectory_sampling_policy, TrajectorySamplingPolicy):
        return trajectory_sampling_policy

    normalized = str(trajectory_sampling_policy).strip().lower()
    aliases = {
        "dense": TrajectorySamplingPolicy.DENSE,
        "adaptive_stride": TrajectorySamplingPolicy.ADAPTIVE_STRIDE,
        "stride": TrajectorySamplingPolicy.ADAPTIVE_STRIDE,
        "adaptive": TrajectorySamplingPolicy.ADAPTIVE_STRIDE,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported trajectory_sampling_policy: {trajectory_sampling_policy}")
    return aliases[normalized]


def normalize_trajectory_rank_policy(
        trajectory_rank_policy: str | TrajectoryRankPolicy,
) -> TrajectoryRankPolicy:
    if isinstance(trajectory_rank_policy, TrajectoryRankPolicy):
        return trajectory_rank_policy

    normalized = str(trajectory_rank_policy).strip().lower()
    aliases = {
        "none": TrajectoryRankPolicy.NONE,
        "explained_dispersion": TrajectoryRankPolicy.EXPLAINED_DISPERSION,
        "explained_variance": TrajectoryRankPolicy.EXPLAINED_DISPERSION,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported trajectory_rank_policy: {trajectory_rank_policy}")
    return aliases[normalized]


def normalize_trajectory_representation_policy(
        trajectory_representation_policy: str | TrajectoryRepresentationPolicy,
) -> TrajectoryRepresentationPolicy:
    if isinstance(trajectory_representation_policy, TrajectoryRepresentationPolicy):
        return trajectory_representation_policy

    normalized = str(trajectory_representation_policy).strip().lower()
    aliases = {
        "none": TrajectoryRepresentationPolicy.NONE,
        "reconstructed": TrajectoryRepresentationPolicy.RECONSTRUCTED,
        "projected": TrajectoryRepresentationPolicy.PROJECTED,
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported trajectory_representation_policy: {trajectory_representation_policy}"
        )
    return aliases[normalized]


def normalize_latent_trajectory_stride_policy(
        latent_trajectory_stride_policy: str | LatentTrajectoryStridePolicy,
) -> LatentTrajectoryStridePolicy:
    if isinstance(latent_trajectory_stride_policy, LatentTrajectoryStridePolicy):
        return latent_trajectory_stride_policy

    normalized = str(latent_trajectory_stride_policy).strip().lower()
    aliases = {
        "dense": LatentTrajectoryStridePolicy.DENSE,
        "adaptive": LatentTrajectoryStridePolicy.ADAPTIVE,
        "adaptive_stride": LatentTrajectoryStridePolicy.ADAPTIVE,
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported latent_trajectory_stride_policy: {latent_trajectory_stride_policy}"
        )
    return aliases[normalized]


def uses_dmd(method: str | OKHSMethod) -> bool:
    return normalize_okhs_method(method) is OKHSMethod.DMD


def uses_direct_kernel_regression(method: str | OKHSMethod) -> bool:
    return normalize_okhs_method(method) in {OKHSMethod.DIRECT, OKHSMethod.OCCUPATION}


def canonical_method_name(method: str | OKHSMethod) -> str:
    return normalize_okhs_method(method).value


def _normalize_trajectories_for_q_selection(data: Any) -> Sequence[Any]:
    if isinstance(data, (list, tuple)):
        return data
    if hasattr(data, "ndim") and getattr(data, "ndim") > 1:
        return list(data)
    return [data]


def resolve_okhs_q(
        q: float,
        q_policy: str | QPolicy,
        trajectories: Iterable[Any],
        q_selector: Any = None,
) -> float:
    normalized_policy = normalize_q_policy(q_policy)
    if normalized_policy is QPolicy.FIXED:
        return q

    selector = q_selector
    if selector is None:
        from fedot_ind.core.operation.transformation.representation.kernel.kernels import DataDrivenQSelector

        selector = DataDrivenQSelector()

    normalized_trajectories = _normalize_trajectories_for_q_selection(trajectories)
    return float(selector.analyze_and_suggest_q(normalized_trajectories, verbose=False))


def _estimate_dominant_period(series: Sequence[Any]) -> int | None:
    values = getattr(series, "reshape", None)
    normalized = series
    if values is not None:
        normalized = series.reshape(-1)
    normalized = [float(value) for value in normalized]
    n = len(normalized)
    if n < 8:
        return None

    centered = [value - sum(normalized) / n for value in normalized]
    variance = sum(value * value for value in centered)
    if variance <= 1e-12:
        return None

    min_lag = max(2, n // 20)
    max_lag = max(min_lag + 1, n // 3)
    best_lag = None
    best_score = 0.0
    for lag in range(min_lag, max_lag + 1):
        numerator = 0.0
        for index in range(n - lag):
            numerator += centered[index] * centered[index + lag]
        score = numerator / variance
        if score > best_score:
            best_score = score
            best_lag = lag
    if best_lag is None or best_score < 0.2:
        return None
    return int(best_lag)


def resolve_okhs_window_size(
        window_size: int | None,
        window_policy: str | WindowPolicy,
        time_series: Sequence[Any],
        forecast_horizon: int,
        min_ratio: float = 0.10,
        max_ratio: float = 0.25,
) -> int:
    return analyze_okhs_window_size(
        window_size=window_size,
        window_policy=window_policy,
        time_series=time_series,
        forecast_horizon=forecast_horizon,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
    )["resolved_window_size"]


def analyze_okhs_window_size(
        window_size: int | None,
        window_policy: str | WindowPolicy,
        time_series: Sequence[Any],
        forecast_horizon: int,
        min_ratio: float = 0.10,
        max_ratio: float = 0.25,
) -> dict[str, Any]:
    normalized_policy = normalize_window_policy(window_policy)
    series_length = len(time_series)
    if series_length <= 2:
        raise ValueError("time_series must contain at least 3 points to resolve window size.")

    max_allowed = max(2, series_length - 1)
    explicit_window = int(window_size) if window_size is not None else None
    if normalized_policy is WindowPolicy.FIXED:
        if explicit_window is None:
            raise ValueError("window_size must be provided when window_policy='fixed'.")
        resolved_window = max(2, min(explicit_window, max_allowed))
        return {
            "window_policy": normalized_policy.value,
            "series_length": series_length,
            "forecast_horizon": forecast_horizon,
            "min_window": 2,
            "max_window": max_allowed,
            "dominant_period": None,
            "candidate_window": explicit_window,
            "resolved_window_size": resolved_window,
            "window_fraction": resolved_window / series_length,
            "trajectory_count": max(series_length - resolved_window, 0),
            "expected_overlap_ratio": (resolved_window - 1) / resolved_window if resolved_window > 0 else 0.0,
        }

    min_window = max(2, int(round(series_length * min_ratio)))
    max_window = max(min_window, int(round(series_length * max_ratio)))
    max_window = min(max_window, max_allowed)
    min_window = min(min_window, max_window)

    dominant_period = _estimate_dominant_period(time_series)
    if normalized_policy is WindowPolicy.ADAPTIVE_CYCLE_AWARE and dominant_period is not None:
        candidate = int(round(1.5 * dominant_period))
    elif dominant_period is not None:
        candidate = dominant_period
    elif explicit_window is not None:
        candidate = explicit_window
    else:
        candidate = int(round(series_length * ((min_ratio + max_ratio) / 2.0)))

    candidate = max(candidate, forecast_horizon + 1)
    resolved_window = max(min_window, min(candidate, max_window))
    return {
        "window_policy": normalized_policy.value,
        "series_length": series_length,
        "forecast_horizon": forecast_horizon,
        "min_window": min_window,
        "max_window": max_window,
        "dominant_period": dominant_period,
        "candidate_window": candidate,
        "resolved_window_size": resolved_window,
        "window_fraction": resolved_window / series_length,
        "trajectory_count": max(series_length - resolved_window, 0),
        "expected_overlap_ratio": (resolved_window - 1) / resolved_window if resolved_window > 0 else 0.0,
    }


def _dense_okhs_trajectory_matrix(time_series: Sequence[Any], window_size: int) -> np.ndarray:
    try:
        dense_matrix = np.asarray(
            build_hankel(time_series=time_series, window_size=window_size).matrix,
            dtype=float,
        )
    except Exception:  # pragma: no cover - fallback for lightweight envs without optional deps
        series = np.asarray(time_series, dtype=float).reshape(-1)
        dense_matrix = np.array(
            [series[index:index + window_size] for index in range(len(series) - window_size + 1)],
            dtype=float,
        )

    # Keep legacy OKHS trajectory count semantics: len(series) - window_size.
    if dense_matrix.shape[0] > 1:
        return dense_matrix[:-1]
    return dense_matrix


def _resolve_adaptive_stride(
        window_size: int,
        series_length: int,
        expected_overlap_ratio: float,
        minimum_trajectory_count: int,
) -> int:
    if series_length <= window_size + 2:
        return 1

    if expected_overlap_ratio < 0.85:
        return 1

    dense_trajectory_count = max(series_length - window_size, 1)
    # Keep adaptive sampling denser for forecasting DMD to preserve short-range patterns.
    base_stride = max(1, int(round(window_size * 0.05)))
    if expected_overlap_ratio >= 0.98:
        base_stride = max(base_stride, int(round(window_size * 0.07)))
    elif expected_overlap_ratio >= 0.95:
        base_stride = max(base_stride, int(round(window_size * 0.06)))

    max_stride_by_count = max(1, dense_trajectory_count // max(minimum_trajectory_count, 1))
    return int(max(1, min(base_stride, max_stride_by_count)))


def _resolve_minimum_selected_rank(
        window_size: int,
        forecast_horizon: int,
        min_dim: int,
        trajectory_rank_value: int | None = None,
) -> int:
    if min_dim <= 0:
        return 0

    explicit_rank = int(trajectory_rank_value) if trajectory_rank_value is not None else None
    if explicit_rank is not None:
        return int(max(2, min(explicit_rank, min_dim)))

    rank_floor = max(
        4,
        int(forecast_horizon),
        int(round(window_size * 0.20)),
    )
    return int(max(2, min(rank_floor, min_dim)))


def _resolve_default_representation_policy(
        window_policy: WindowPolicy,
        representation_policy: str | TrajectoryRepresentationPolicy | None,
) -> TrajectoryRepresentationPolicy:
    if representation_policy is not None:
        return normalize_trajectory_representation_policy(representation_policy)
    if window_policy is WindowPolicy.FIXED:
        return TrajectoryRepresentationPolicy.NONE
    return TrajectoryRepresentationPolicy.PROJECTED


def _resolve_latent_trajectory_stride(
        latent_state_count: int,
        latent_window_size: int,
        forecast_horizon: int,
        selected_rank: int,
        latent_trajectory_stride_policy: str | LatentTrajectoryStridePolicy,
        latent_trajectory_stride: int | None = None,
) -> tuple[int, int]:
    dense_latent_trajectory_count = max(latent_state_count - latent_window_size, 1)
    normalized_policy = normalize_latent_trajectory_stride_policy(latent_trajectory_stride_policy)
    if normalized_policy is LatentTrajectoryStridePolicy.DENSE:
        stride = int(max(1, latent_trajectory_stride or 1))
        return stride, dense_latent_trajectory_count

    if latent_trajectory_stride is not None:
        stride = int(max(1, latent_trajectory_stride))
        return stride, dense_latent_trajectory_count

    target_overlap_ratio = 0.80
    stride_from_overlap = max(1, int(round(latent_window_size * (1.0 - target_overlap_ratio))))
    minimum_effective_trajectory_count = max(
        8,
        forecast_horizon + 4,
        selected_rank + forecast_horizon,
    )
    max_stride_by_count = max(1, dense_latent_trajectory_count // max(minimum_effective_trajectory_count, 1))
    stride = int(max(1, min(stride_from_overlap, max_stride_by_count)))
    return stride, dense_latent_trajectory_count


def analyze_okhs_trajectory_preprocessing(
        time_series: Sequence[Any],
        window_size: int,
        window_policy: str | WindowPolicy,
        forecast_horizon: int,
        trajectory_sampling_policy: str | TrajectorySamplingPolicy = TrajectorySamplingPolicy.DENSE,
        trajectory_rank_policy: str | TrajectoryRankPolicy = TrajectoryRankPolicy.EXPLAINED_DISPERSION,
        trajectory_rank_value: int | None = None,
        trajectory_representation_policy: str | TrajectoryRepresentationPolicy | None = None,
) -> dict[str, Any]:
    window_diagnostics = analyze_okhs_window_size(
        window_size=window_size,
        window_policy=window_policy,
        time_series=time_series,
        forecast_horizon=forecast_horizon,
    )
    normalized_window_policy = normalize_window_policy(window_policy)
    normalized_sampling_policy = normalize_trajectory_sampling_policy(trajectory_sampling_policy)
    normalized_rank_policy = normalize_trajectory_rank_policy(trajectory_rank_policy)
    normalized_representation_policy = _resolve_default_representation_policy(
        window_policy=normalized_window_policy,
        representation_policy=trajectory_representation_policy,
    )
    adaptive_window_enabled = normalized_window_policy is not WindowPolicy.FIXED

    if not adaptive_window_enabled:
        normalized_sampling_policy = TrajectorySamplingPolicy.DENSE
        normalized_rank_policy = TrajectoryRankPolicy.NONE
        normalized_representation_policy = TrajectoryRepresentationPolicy.NONE

    minimum_trajectory_count = max(12, forecast_horizon * 2,
                                   int(round(window_diagnostics["resolved_window_size"] * 0.75)))
    effective_stride = 1
    if normalized_sampling_policy is TrajectorySamplingPolicy.ADAPTIVE_STRIDE:
        effective_stride = _resolve_adaptive_stride(
            window_size=window_diagnostics["resolved_window_size"],
            series_length=window_diagnostics["series_length"],
            expected_overlap_ratio=window_diagnostics["expected_overlap_ratio"],
            minimum_trajectory_count=minimum_trajectory_count,
        )

    dense_trajectory_count = max(window_diagnostics["trajectory_count"], 0)
    effective_trajectory_count = dense_trajectory_count
    if dense_trajectory_count > 0 and effective_stride > 1:
        effective_trajectory_count = int(len(range(0, dense_trajectory_count, effective_stride)))

    return {
        "enabled": adaptive_window_enabled,
        "window_policy": normalized_window_policy.value,
        "window_size": window_diagnostics["resolved_window_size"],
        "series_length": window_diagnostics["series_length"],
        "forecast_horizon": forecast_horizon,
        "trajectory_sampling_policy": normalized_sampling_policy.value,
        "trajectory_rank_policy": normalized_rank_policy.value,
        "trajectory_rank_value": trajectory_rank_value,
        "trajectory_representation_policy": normalized_representation_policy.value,
        "effective_stride": effective_stride,
        "minimum_trajectory_count": minimum_trajectory_count,
        "dense_trajectory_count": dense_trajectory_count,
        "effective_trajectory_count": effective_trajectory_count,
        "recommended_min_selected_rank": _resolve_minimum_selected_rank(
            window_size=window_diagnostics["resolved_window_size"],
            forecast_horizon=forecast_horizon,
            min_dim=min(window_diagnostics["resolved_window_size"], max(effective_trajectory_count, 1)),
            trajectory_rank_value=trajectory_rank_value,
        ),
        "expected_overlap_ratio": window_diagnostics["expected_overlap_ratio"],
        "window_fraction": window_diagnostics["window_fraction"],
        "window_diagnostics": window_diagnostics,
    }


def apply_trajectory_rank_regularization(
        trajectory_matrix: np.ndarray,
        trajectory_rank_policy: str | TrajectoryRankPolicy,
        trajectory_rank_value: int | None = None,
        min_selected_rank: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    normalized_rank_policy = normalize_trajectory_rank_policy(trajectory_rank_policy)
    matrix = np.asarray(trajectory_matrix, dtype=float)
    min_dim = int(min(matrix.shape)) if matrix.size else 0

    if normalized_rank_policy is TrajectoryRankPolicy.NONE or matrix.ndim != 2 or min_dim == 0:
        return matrix, {
            "trajectory_rank_policy": normalized_rank_policy.value,
            "selected_rank": min_dim,
            "requested_rank_floor": min_selected_rank,
            "applied_rank_floor": min_selected_rank,
            "rank_floor_applied": False,
            "explained_variance_retained": 1.0,
            "compression_ratio": 1.0,
            "singular_values": np.array([], dtype=float) if min_dim == 0 else None,
        }

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0:
        return matrix, {
            "trajectory_rank_policy": normalized_rank_policy.value,
            "selected_rank": 0,
            "requested_rank_floor": min_selected_rank,
            "applied_rank_floor": min_selected_rank,
            "rank_floor_applied": False,
            "explained_variance_retained": 1.0,
            "compression_ratio": 1.0,
            "singular_values": singular_values,
        }

    if normalized_rank_policy is TrajectoryRankPolicy.EXPLAINED_DISPERSION:
        try:
            from fedot_ind.core.operation.transformation.regularization.spectrum import (
                sv_to_explained_variance_ratio,
            )

            selected_rank = sv_to_explained_variance_ratio(singular_values)
        except Exception:  # pragma: no cover - lightweight fallback for reduced test envs
            normalized_sv = np.abs(singular_values)
            total = float(np.sum(normalized_sv))
            if total <= 0:
                selected_rank = min_dim
            else:
                variance = normalized_sv / total * 100.0
                selected_rank = int(np.sum(variance > 3.0))
                if selected_rank == 0:
                    selected_rank = min(2, min_dim)
    else:
        selected_rank = trajectory_rank_value or min_dim

    applied_rank_floor = int(max(2, min(min_selected_rank, min_dim))) if min_selected_rank is not None else 2
    raw_selected_rank = int(max(2, min(selected_rank, min_dim)))
    selected_rank = int(max(raw_selected_rank, applied_rank_floor))
    truncated = truncate_rank(
        matrix=matrix,
        rank=selected_rank,
        explained_variance=0.95,
        min_rank=max(1, min_selected_rank or 1),
    )
    approximated = truncated.reconstructed_matrix
    retained = float(truncated.explained_variance_retained)

    return approximated, {
        "trajectory_rank_policy": normalized_rank_policy.value,
        "selected_rank": selected_rank,
        "raw_selected_rank": raw_selected_rank,
        "requested_rank_floor": min_selected_rank,
        "applied_rank_floor": applied_rank_floor,
        "rank_floor_applied": selected_rank > raw_selected_rank,
        "explained_variance_retained": retained,
        "compression_ratio": selected_rank / min_dim if min_dim > 0 else 1.0,
        "singular_values": truncated.singular_values,
    }


def _build_projected_trajectory_windows(
        latent_states: np.ndarray,
        preferred_window_size: int,
        minimum_trajectory_count: int,
        forecast_horizon: int,
        selected_rank: int,
        latent_trajectory_stride_policy: str | LatentTrajectoryStridePolicy = LatentTrajectoryStridePolicy.ADAPTIVE,
        latent_trajectory_stride: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    state_matrix = np.asarray(latent_states, dtype=float)
    if state_matrix.ndim != 2:
        raise ValueError("latent_states must have shape (n_states, n_features).")

    n_states = state_matrix.shape[0]
    if n_states < 3:
        raise ValueError("Projected trajectory construction requires at least 3 latent states.")

    max_window_by_count = max(2, n_states - max(minimum_trajectory_count, 1))
    resolved_window_size = min(preferred_window_size, max_window_by_count, n_states - 1)
    resolved_window_size = max(forecast_horizon + 1, resolved_window_size)
    resolved_window_size = min(resolved_window_size, n_states - 1)

    latent_stride, dense_latent_trajectory_count = _resolve_latent_trajectory_stride(
        latent_state_count=n_states,
        latent_window_size=resolved_window_size,
        forecast_horizon=forecast_horizon,
        selected_rank=selected_rank,
        latent_trajectory_stride_policy=latent_trajectory_stride_policy,
        latent_trajectory_stride=latent_trajectory_stride,
    )
    trajectories = np.array(
        [
            state_matrix[index:index + resolved_window_size]
            for index in range(0, n_states - resolved_window_size, latent_stride)
        ],
        dtype=float,
    )
    if trajectories.size == 0:
        trajectories = state_matrix[-resolved_window_size:].reshape(1, resolved_window_size, state_matrix.shape[1])
    effective_latent_trajectory_count = int(trajectories.shape[0])
    latent_overlap_ratio = (
        max(0.0, (resolved_window_size - latent_stride) / resolved_window_size)
        if resolved_window_size > 0 else 0.0
    )
    return trajectories, {
        "latent_window_size": int(resolved_window_size),
        "latent_stride": int(latent_stride),
        "latent_trajectory_stride_policy": normalize_latent_trajectory_stride_policy(
            latent_trajectory_stride_policy
        ).value,
        "dense_latent_trajectory_count": int(dense_latent_trajectory_count),
        "effective_latent_trajectory_count": effective_latent_trajectory_count,
        "latent_overlap_ratio": float(latent_overlap_ratio),
        "latent_state_count": int(n_states),
    }


def build_okhs_projected_state_sequence(
        time_series: Sequence[Any],
        window_size: int,
        effective_stride: int,
        basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dense_matrix = _dense_okhs_trajectory_matrix(
        time_series=time_series,
        window_size=window_size,
    )
    sampled_matrix = dense_matrix[::effective_stride] if effective_stride > 1 and dense_matrix.shape[
        0] > 0 else dense_matrix
    basis_matrix = np.asarray(basis, dtype=float)
    latent_states = sampled_matrix @ basis_matrix
    return sampled_matrix, latent_states


def build_okhs_trajectory_representation(
        time_series: Sequence[Any],
        window_size: int,
        window_policy: str | WindowPolicy,
        forecast_horizon: int,
        trajectory_sampling_policy: str | TrajectorySamplingPolicy = TrajectorySamplingPolicy.ADAPTIVE_STRIDE,
        trajectory_rank_policy: str | TrajectoryRankPolicy = TrajectoryRankPolicy.EXPLAINED_DISPERSION,
        trajectory_rank_value: int | None = None,
        trajectory_representation_policy: str | TrajectoryRepresentationPolicy | None = None,
        latent_trajectory_stride_policy: str | LatentTrajectoryStridePolicy = LatentTrajectoryStridePolicy.ADAPTIVE,
        latent_trajectory_stride: int | None = None,
) -> dict[str, Any]:
    preprocessing = analyze_okhs_trajectory_preprocessing(
        time_series=time_series,
        window_size=window_size,
        window_policy=window_policy,
        forecast_horizon=forecast_horizon,
        trajectory_sampling_policy=trajectory_sampling_policy,
        trajectory_rank_policy=trajectory_rank_policy,
        trajectory_rank_value=trajectory_rank_value,
        trajectory_representation_policy=trajectory_representation_policy,
    )

    dense_matrix = _dense_okhs_trajectory_matrix(
        time_series=time_series,
        window_size=preprocessing["window_size"],
    )
    sampled_matrix = dense_matrix
    if preprocessing["effective_stride"] > 1 and dense_matrix.shape[0] > 0:
        sampled_matrix = dense_matrix[::preprocessing["effective_stride"]]

    matrix_after, rank_diagnostics = apply_trajectory_rank_regularization(
        trajectory_matrix=sampled_matrix,
        trajectory_rank_policy=preprocessing["trajectory_rank_policy"],
        trajectory_rank_value=preprocessing["trajectory_rank_value"],
        min_selected_rank=preprocessing["recommended_min_selected_rank"],
    )
    selected_rank = int(rank_diagnostics["selected_rank"]) if sampled_matrix.size else 0
    normalized_representation_policy = normalize_trajectory_representation_policy(
        preprocessing["trajectory_representation_policy"]
    )

    projection_metadata = {
        "representation_policy": normalized_representation_policy.value,
        "basis_shape": None,
        "projected_shape": None,
        "decode_supported": False,
        "latent_window_size": None,
        "latent_stride": None,
        "latent_trajectory_stride_policy": None,
        "dense_latent_trajectory_count": None,
        "effective_latent_trajectory_count": None,
        "latent_overlap_ratio": None,
        "latent_state_count": None,
        "decode_reconstruction_error": None,
    }
    projection_runtime = None
    training_matrix = matrix_after
    trajectory_matrix_shape_after = tuple(int(value) for value in matrix_after.shape)

    if normalized_representation_policy is TrajectoryRepresentationPolicy.PROJECTED and sampled_matrix.size:
        truncated = truncate_rank(
            matrix=sampled_matrix,
            rank=selected_rank,
            explained_variance=0.95,
            min_rank=max(1, selected_rank),
        )
        basis = truncated.basis
        latent_states = truncated.projected_states
        projected_trajectories, latent_window_diagnostics = _build_projected_trajectory_windows(
            latent_states=latent_states,
            preferred_window_size=preprocessing["window_size"],
            minimum_trajectory_count=preprocessing["minimum_trajectory_count"],
            forecast_horizon=forecast_horizon,
            selected_rank=selected_rank,
            latent_trajectory_stride_policy=latent_trajectory_stride_policy,
            latent_trajectory_stride=latent_trajectory_stride,
        )
        decoded_matrix = latent_states @ basis.T
        reconstruction_error = float(
            np.sqrt(np.mean((decoded_matrix - matrix_after) ** 2))
        ) if matrix_after.size else 0.0
        projection_metadata = {
            "representation_policy": normalized_representation_policy.value,
            "basis_shape": tuple(int(value) for value in basis.shape),
            "projected_shape": tuple(int(value) for value in latent_states.shape),
            "decode_supported": True,
            **latent_window_diagnostics,
            "decode_reconstruction_error": reconstruction_error,
        }
        projection_runtime = {
            "basis": basis,
            "latent_state_matrix": latent_states,
            "sampled_matrix": matrix_after,
            "latent_window_size": int(latent_window_diagnostics["latent_window_size"]),
        }
        training_matrix = projected_trajectories
        trajectory_matrix_shape_after = tuple(int(value) for value in projected_trajectories.shape)
    elif normalized_representation_policy is TrajectoryRepresentationPolicy.RECONSTRUCTED:
        projection_metadata = {
            "representation_policy": normalized_representation_policy.value,
            "basis_shape": tuple(
                int(value) for value in (sampled_matrix.shape[1], selected_rank)) if sampled_matrix.size else None,
            "projected_shape": tuple(
                int(value) for value in (sampled_matrix.shape[0], selected_rank)) if sampled_matrix.size else None,
            "decode_supported": True,
            "latent_window_size": None,
            "decode_reconstruction_error": 0.0,
        }

    diagnostics = {
        **preprocessing,
        "trajectory_matrix_shape_before": tuple(int(value) for value in sampled_matrix.shape),
        "trajectory_matrix_shape_after": trajectory_matrix_shape_after,
        "effective_trajectory_count": int(training_matrix.shape[0]) if np.asarray(training_matrix).ndim >= 2 else 0,
        **rank_diagnostics,
        **projection_metadata,
    }
    return {
        "training_matrix": training_matrix,
        "trajectory_preprocessing": diagnostics,
        "projection_metadata": projection_metadata,
        "projection_runtime": projection_runtime,
    }


def build_okhs_trajectory_matrix(
        time_series: Sequence[Any],
        window_size: int,
        window_policy: str | WindowPolicy,
        forecast_horizon: int,
        trajectory_sampling_policy: str | TrajectorySamplingPolicy = TrajectorySamplingPolicy.ADAPTIVE_STRIDE,
        trajectory_rank_policy: str | TrajectoryRankPolicy = TrajectoryRankPolicy.EXPLAINED_DISPERSION,
        trajectory_rank_value: int | None = None,
        trajectory_representation_policy: str | TrajectoryRepresentationPolicy | None = TrajectoryRepresentationPolicy.RECONSTRUCTED,
) -> tuple[np.ndarray, dict[str, Any]]:
    representation = build_okhs_trajectory_representation(
        time_series=time_series,
        window_size=window_size,
        window_policy=window_policy,
        forecast_horizon=forecast_horizon,
        trajectory_sampling_policy=trajectory_sampling_policy,
        trajectory_rank_policy=trajectory_rank_policy,
        trajectory_rank_value=trajectory_rank_value,
        trajectory_representation_policy=trajectory_representation_policy,
    )
    return representation["training_matrix"], representation["trajectory_preprocessing"]
