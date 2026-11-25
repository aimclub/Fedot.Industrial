from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..trajectory import estimate_window


@dataclass(frozen=True)
class RegimeDiagnosticsResult:
    """Compact statistical description used by forecasting regime routing."""

    series_length: int
    dominant_period: int | None
    acf_decay_rate: float
    spectral_concentration: float
    spectral_flatness: float
    local_linearity_score: float
    switching_score: float
    regime_hint: str

    def to_dict(self) -> dict[str, float | int | str | None]:
        """Serialize regime diagnostics into primitive Python values."""
        return asdict(self)


def _autocorrelations(centered: np.ndarray, denominator: float, stop: int) -> np.ndarray:
    """Return normalized linear autocorrelations for lags 0 through stop - 1."""
    if denominator <= 1e-12:
        return np.zeros(stop)
    length = 1 << (2 * len(centered) - 2).bit_length()
    transformed = np.fft.rfft(centered, n=length)
    power = transformed.real ** 2 + transformed.imag ** 2
    return np.fft.irfft(power, n=length)[:stop] / denominator


def _dominant_period(centered: np.ndarray, denominator: float, scores: np.ndarray) -> int | None:
    if not scores.size:
        return None
    best = int(np.argmax(scores))
    tolerance = 8 * np.finfo(float).eps * len(centered).bit_length()
    if not scores[best] >= 0.2 - tolerance:
        return None
    near_peak = np.flatnonzero(scores[best] - scores <= tolerance)
    if near_peak.size == 1 and scores[best] > 0.2 + tolerance:
        return best + 2

    # Use the direct calculation only when roundoff could change the chosen lag or cross the cutoff.
    best_score = 0.0
    best_lag = None
    for index in near_peak:
        lag = int(index) + 2
        score = float(np.sum(centered[:-lag] * centered[lag:]) / denominator)
        if score > best_score:
            best_lag, best_score = lag, score
    return best_lag if best_score >= 0.2 else None


def analyze_regime_diagnostics(time_series: np.ndarray, window_size: int | None = None) -> RegimeDiagnosticsResult:
    """Compute coarse regime indicators from a raw time series."""
    values = np.asarray(time_series, dtype=float).reshape(-1)
    if values.size < 6:
        return RegimeDiagnosticsResult(
            series_length=int(values.size),
            dominant_period=None,
            acf_decay_rate=0.0,
            spectral_concentration=0.0,
            spectral_flatness=1.0,
            local_linearity_score=0.0,
            switching_score=0.0,
            regime_hint='insufficient_history',
        )

    resolved_window = window_size or estimate_window(len(values), forecast_horizon=max(1, len(values) // 20))
    centered = values - np.mean(values)
    denominator = float(np.sum(centered ** 2))
    decay_stop = max(1, min(resolved_window, len(values) // 2, 6))
    period_stop = max(3, len(values) // 3) if len(values) >= 8 else 2
    acf = _autocorrelations(centered, denominator, max(decay_stop, period_stop))
    decay = acf[1:decay_stop]
    acf_decay_rate = float(np.mean(np.abs(decay))) if decay.size else 0.0
    dominant_period = _dominant_period(centered, denominator, acf[2:period_stop])

    spectrum = np.abs(np.fft.rfft(centered))
    if spectrum.size <= 1:
        spectral_concentration = 0.0
        spectral_flatness = 1.0
    else:
        usable = spectrum[1:]
        total = float(np.sum(usable))
        spectral_concentration = float(np.max(usable) / total) if total > 0 else 0.0
        safe = np.maximum(usable, 1e-12)
        spectral_flatness = float(np.exp(np.mean(np.log(safe))) / np.mean(safe))

    local_window = max(4, min(resolved_window, len(values) // 3))
    local_errors = []
    for start in range(0, len(values) - local_window + 1, max(1, local_window // 3)):
        segment = values[start:start + local_window]
        index = np.arange(len(segment), dtype=float)
        slope, intercept = np.polyfit(index, segment, deg=1)
        fitted = intercept + slope * index
        baseline = np.var(segment) + 1e-12
        local_errors.append(float(np.mean((segment - fitted) ** 2) / baseline))
    local_linearity_score = float(1.0 / (1.0 + np.mean(local_errors))) if local_errors else 0.0

    gradients = np.diff(values)
    switching_score = float(np.mean(np.abs(np.diff(np.sign(gradients))))) if gradients.size > 1 else 0.0

    if dominant_period is not None and spectral_concentration > 0.25:
        regime_hint = 'periodic'
    elif switching_score > 0.6:
        regime_hint = 'switching'
    elif local_linearity_score > 0.55:
        regime_hint = 'locally_linear'
    else:
        regime_hint = 'weak_structure'

    return RegimeDiagnosticsResult(
        series_length=int(len(values)),
        dominant_period=dominant_period,
        acf_decay_rate=float(acf_decay_rate),
        spectral_concentration=float(spectral_concentration),
        spectral_flatness=float(spectral_flatness),
        local_linearity_score=float(local_linearity_score),
        switching_score=float(switching_score),
        regime_hint=regime_hint,
    )
