from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from statsmodels.tsa.stattools import pacf

from benchmark.v2.core import ForecastingSeriesRecord
from .entropy import compute_approximate_entropy
from .entropy import compute_permutation_entropy
from .entropy import compute_sample_entropy
from .entropy import compute_spectral_entropy
from .entropy import compute_svd_entropy


@dataclass
class ForecastabilityReport:
    series_id: str
    length: int
    adi: float | None = None
    cv2: float | None = None
    is_intermittent: bool = False
    acf_decay: float | None = None
    pacf_significant: int = 0
    dominant_lag: int | None = None
    sample_entropy: float | None = None
    approx_entropy: float | None = None
    permutation_entropy: float | None = None
    spectral_entropy: float | None = None
    svd_entropy: float | None = None
    dominant_frequency: float | None = None
    hurst: float | None = None
    lyapunov: float | None = None
    n_imfs: int = 0
    mode_energies: list[float] = field(default_factory=list)
    dominant_mode: int | None = None
    regime_hint: str = "unknown"
    predictability_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "length": self.length,
            "adi": self.adi,
            "cv2": self.cv2,
            "is_intermittent": self.is_intermittent,
            "acf_decay": self.acf_decay,
            "pacf_significant": self.pacf_significant,
            "dominant_lag": self.dominant_lag,
            "sample_entropy": self.sample_entropy,
            "approx_entropy": self.approx_entropy,
            "permutation_entropy": self.permutation_entropy,
            "spectral_entropy": self.spectral_entropy,
            "svd_entropy": self.svd_entropy,
            "dominant_frequency": self.dominant_frequency,
            "hurst": self.hurst,
            "lyapunov": self.lyapunov,
            "n_imfs": self.n_imfs,
            "mode_energies": self.mode_energies,
            "dominant_mode": self.dominant_mode,
            "regime_hint": self.regime_hint,
            "predictability_score": self.predictability_score,
            **self.metadata,
        }


def _is_finite(value: float | None) -> bool:
    return value is not None and np.isfinite(value)


def compute_adi_cv(values: np.ndarray) -> tuple[float, float, bool]:
    values = np.asarray(values, dtype=float)
    non_zero = values[values > 0]

    if len(non_zero) < 2:
        return float("nan"), float("nan"), False

    indices = np.where(values > 0)[0]
    intervals = np.diff(indices)

    adi = float(np.mean(intervals)) if len(intervals) > 0 else float("inf")

    mean = float(np.mean(non_zero))
    std = float(np.std(non_zero))

    cv2 = (std / mean) ** 2 if mean > 0 else float("inf")
    intermittent = adi > 1.32 and cv2 > 0.49

    return float(adi), float(cv2), intermittent


def compute_acf_pacf(values: np.ndarray, max_lags: int = 40) -> tuple[float, int, int | None]:
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n < 8:
        return float("nan"), 0, None

    max_lags = min(max_lags, max(2, n // 4))

    centered = values - np.mean(values)

    acf = np.correlate(centered, centered, mode="full")
    acf = acf[len(acf) // 2:]

    if acf[0] == 0:
        return float("nan"), 0, None

    acf = acf / acf[0]
    acf = acf[:max_lags + 1]

    acf_decay = float(np.mean(np.abs(acf[1:min(6, len(acf))])))

    try:
        pacf_values = pacf(values, nlags=max_lags, method="ywadjusted")
        threshold = 1.96 / np.sqrt(n)
        significant_lags = int(np.sum(np.abs(pacf_values[1:]) > threshold))
    except Exception:
        significant_lags = 0

    dominant_lag = int(np.argmax(np.abs(acf[1:])) + 1) if len(acf) > 1 else None

    return acf_decay, significant_lags, dominant_lag


def compute_dominant_frequency(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)

    if len(values) < 8:
        return float("nan")

    centered = values - np.mean(values)

    spectrum = np.fft.rfft(centered)
    power = np.abs(spectrum) ** 2

    if len(power) <= 1:
        return float("nan")

    power[0] = 0.0

    dominant = int(np.argmax(power))

    if dominant == 0:
        return float("nan")

    frequencies = np.fft.rfftfreq(len(values))

    return float(frequencies[dominant])


def compute_hurst(values: np.ndarray, max_lag: int = 100) -> float:
    values = np.asarray(values, dtype=float)

    if len(values) < 32:
        return 0.5

    upper = min(max_lag, len(values) // 2)

    if upper <= 4:
        return 0.5

    lags = np.arange(2, upper)

    tau = []

    for lag in lags:
        diff = values[lag:] - values[:-lag]
        sigma = np.std(diff)

        if sigma > 0:
            tau.append(np.sqrt(sigma))

    if len(tau) < 4:
        return 0.5

    tau = np.asarray(tau)

    hurst = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)[0] * 2.0

    return float(np.clip(hurst, 0.0, 1.0))


def compute_lyapunov(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)

    if len(values) < 100:
        return 0.0

    divergences = []

    for lag in range(1, min(20, len(values) // 10)):
        diff = np.abs(values[lag:] - values[:-lag])
        divergences.append(np.mean(diff))

    divergences = np.asarray(divergences)
    valid = divergences > 0

    if np.sum(valid) < 3:
        return 0.0

    x = np.arange(1, len(divergences) + 1)[valid]
    y = np.log(divergences[valid] + 1e-12)
    slope = np.polyfit(x, y, 1)[0]

    return float(np.clip(slope, 0.0, 2.0))


def detect_regime(report: ForecastabilityReport) -> str:
    if report.is_intermittent:
        return "intermittent"

    scores = {
        "trend": 0.0,
        "periodic": 0.0,
        "chaotic": 0.0,
        "stationary": 0.0,
    }

    if _is_finite(report.hurst):
        scores["trend"] += max(0.0, report.hurst - 0.5)
        scores["stationary"] += max(0.0, 0.6 - report.hurst)

    if _is_finite(report.spectral_entropy):
        scores["periodic"] += 1.0 - report.spectral_entropy
        scores["chaotic"] += report.spectral_entropy

    if _is_finite(report.sample_entropy):
        entropy_score = min(1.0, report.sample_entropy / 2.0)
        scores["chaotic"] += entropy_score
        scores["stationary"] += 1.0 - entropy_score

    if _is_finite(report.lyapunov):
        scores["chaotic"] += min(1.0, report.lyapunov)

    return max(scores.items(), key=lambda item: item[1])[0]


def compute_predictability_score(report: ForecastabilityReport) -> float:
    components = []

    if _is_finite(report.hurst):
        components.append(np.clip(report.hurst, 0.0, 1.0))

    if _is_finite(report.spectral_entropy):
        components.append(1.0 - np.clip(report.spectral_entropy, 0.0, 1.0))

    if _is_finite(report.sample_entropy):
        entropy = min(1.0, report.sample_entropy / 2.0)
        components.append(1.0 - entropy)

    if _is_finite(report.svd_entropy):
        components.append(1.0 - np.clip(report.svd_entropy, 0.0, 1.0))

    if _is_finite(report.acf_decay):
        components.append(np.clip(report.acf_decay, 0.0, 1.0))

    if not components:
        return 0.5

    return float(np.mean(components))


def analyze_forecastability(series_record: ForecastingSeriesRecord, include_emd: bool = False) -> ForecastabilityReport:
    values = np.asarray(series_record.train_values, dtype=float)

    report = ForecastabilityReport(
        series_id=series_record.series_id,
        length=len(values),
    )

    report.adi, report.cv2, report.is_intermittent = compute_adi_cv(values)
    report.acf_decay, report.pacf_significant, report.dominant_lag = compute_acf_pacf(values)
    report.sample_entropy = compute_sample_entropy(values)
    report.approx_entropy = compute_approximate_entropy(values)
    report.permutation_entropy = compute_permutation_entropy(values)
    report.spectral_entropy = compute_spectral_entropy(values)
    report.svd_entropy = compute_svd_entropy(values)
    report.dominant_frequency = compute_dominant_frequency(values)
    report.hurst = compute_hurst(values)
    report.lyapunov = compute_lyapunov(values)

    if include_emd:
        from .decomposition import compute_emd_diagnostics

        diagnostics = compute_emd_diagnostics(values)

        report.n_imfs = diagnostics.get("n_imfs", 0)
        report.mode_energies = diagnostics.get("mode_energies", [])
        report.dominant_mode = diagnostics.get("dominant_mode")

    report.regime_hint = detect_regime(report)
    report.predictability_score = compute_predictability_score(report)

    return report