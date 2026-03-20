from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Sequence


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
