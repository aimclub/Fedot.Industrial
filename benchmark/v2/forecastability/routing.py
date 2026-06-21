from __future__ import annotations

from typing import Any
from collections import defaultdict

from .diagnostics import ForecastabilityReport

def _is_finite(value: float | None) -> bool:
    return value is not None and value == value


def _should_use_naive_first(report: ForecastabilityReport) -> tuple[bool, float]:

    if _is_finite(report.predictability_score) and report.predictability_score < 0.15:
        return True, 0.95

    if report.length < 20:
        if _is_finite(report.acf_decay) and report.acf_decay < 0.2:
            return True, 0.85

    return False, 0.0


def _score_hurst(report: ForecastabilityReport) -> dict[str, float]:
    scores = defaultdict(float)

    if not _is_finite(report.hurst):
        return scores

    if report.hurst > 0.65:
        scores["classical_baseline"] += 0.12
        scores["automl"] += 0.15
        scores["internal_industrial"] += 0.08
        if report.hurst > 0.75:
            scores["automl"] += 0.08
            scores["foundation"] += 0.04

    elif report.hurst < 0.35:
        scores["classical_baseline"] += 0.08
        scores["internal_industrial"] += 0.10

    else:
        scores["classical_baseline"] += 0.03

    return scores


def _score_entropy(report: ForecastabilityReport) -> dict[str, float]:
    scores = defaultdict(float)

    if _is_finite(report.sample_entropy):
        if report.sample_entropy > 2.0:
            scores["naive_baseline"] += 0.30
            scores["foundation"] -= 0.10
        elif report.sample_entropy > 1.5:
            scores["foundation"] += 0.12
            scores["internal_industrial"] += 0.08
        elif report.sample_entropy < 0.5:
            scores["classical_baseline"] += 0.12
            scores["automl"] += 0.08

    return scores


def _score_spectral(report: ForecastabilityReport) -> dict[str, float]:
    scores = defaultdict(float)

    if not _is_finite(report.spectral_entropy):
        return scores

    if report.spectral_entropy < 0.3:
        scores["classical_baseline"] += 0.20
        scores["internal_industrial"] += 0.08

    elif report.spectral_entropy > 0.75:
        scores["naive_baseline"] += 0.20
        scores["foundation"] += 0.08

    else:
        scores["automl"] += 0.08

    return scores


def _score_acf_pacf(report: ForecastabilityReport) -> dict[str, float]:
    scores = defaultdict(float)

    if not _is_finite(report.acf_decay):
        return scores

    if report.acf_decay > 0.7:
        scores["automl"] += 0.12
        scores["classical_baseline"] += 0.08

    elif report.acf_decay < 0.2:
        scores["classical_baseline"] += 0.08

    return scores


def _score_intermittent(report: ForecastabilityReport) -> dict[str, float]:
    scores = defaultdict(float)

    if not report.is_intermittent:
        return scores

    scores["classical_baseline"] += 0.20

    if _is_finite(report.adi) and _is_finite(report.cv2):
        if report.adi > 3.0:
            scores["classical_baseline"] += 0.10
        if report.cv2 > 1.5:
            scores["classical_baseline"] += 0.05

    return scores


def _score_emd(report: ForecastabilityReport) -> dict[str, float]:
    scores = defaultdict(float)

    if not report.mode_energies or len(report.mode_energies) == 0:
        return scores

    if report.n_imfs >= 5:
        scores["foundation"] += 0.08
        scores["internal_industrial"] += 0.06

    if report.dominant_mode is not None and report.mode_energies:
        dominant_energy = report.mode_energies[report.dominant_mode]
        if dominant_energy > 0.7:
            scores["classical_baseline"] += 0.06

    return scores


def _score_predictability(report: ForecastabilityReport) -> dict[str, float]:
    scores = defaultdict(float)

    if not _is_finite(report.predictability_score):
        return scores

    score = report.predictability_score

    if score > 0.7:
        scores["classical_baseline"] += 0.15
        scores["automl"] += 0.08
    elif score > 0.4:
        scores["automl"] += 0.08
        scores["internal_industrial"] += 0.04
    elif score > 0.15:
        scores["foundation"] += 0.10
        scores["internal_industrial"] += 0.08
    else:
        scores["naive_baseline"] += 0.35

    return scores


def _score_lyapunov(report: ForecastabilityReport) -> dict[str, float]:
    scores = defaultdict(float)

    if not _is_finite(report.lyapunov):
        return scores

    if report.lyapunov > 0.3:
        scores["foundation"] += 0.10
        scores["internal_industrial"] += 0.08
    elif report.lyapunov > 0.1:
        scores["foundation"] += 0.04
        scores["internal_industrial"] += 0.03

    return scores


def _score_series_length(report: ForecastabilityReport) -> dict[str, float]:
    scores = defaultdict(float)

    if report.length < 20:
        scores["classical_baseline"] += 0.15
        scores["foundation"] -= 0.05
        scores["automl"] -= 0.05
    elif report.length < 50:
        scores["classical_baseline"] += 0.06
    elif report.length > 200:
        scores["foundation"] += 0.08
        scores["automl"] += 0.06
        scores["internal_industrial"] += 0.04

    return scores


def recommend_model_family(report: ForecastabilityReport) -> tuple[str, float]:
    use_naive, naive_conf = _should_use_naive_first(report)
    if use_naive:
        return "naive_baseline", naive_conf

    weights = {
        "naive_baseline": 0.10,
        "classical_baseline": 0.35,
        "automl": 0.22,
        "internal_industrial": 0.18,
        "foundation": 0.15,
    }

    scorers = [
        _score_hurst,
        _score_entropy,
        _score_spectral,
        _score_acf_pacf,
        _score_intermittent,
        _score_emd,
        _score_predictability,
        _score_lyapunov,
        _score_series_length,
    ]

    for scorer in scorers:
        scores = scorer(report)
        for family, delta in scores.items():
            weights[family] = max(0.0, weights.get(family, 0.0) + delta)

    model_family, confidence = max(weights.items(), key=lambda x: x[1])

    if len(weights) > 1:
        sorted_weights = sorted(weights.values(), reverse=True)
        if sorted_weights[0] > 0:
            confidence = (sorted_weights[0] - sorted_weights[1]) / sorted_weights[0]
            confidence = min(0.95, max(0.25, confidence))
        else:
            confidence = 0.5
    else:
        confidence = 0.5

    return model_family, float(confidence)


def recommend_specific_model(report: ForecastabilityReport) -> tuple[str, float]:
    pass


def get_routing_hints(report: ForecastabilityReport) -> dict[str, Any]:

    model_family, family_confidence = recommend_model_family(report)
    specific_model, model_confidence = recommend_specific_model(report)

    return {
        "recommended_family": model_family,
        "confidence": family_confidence,
        "recommended_specific_model": specific_model,
        "specific_model_confidence": model_confidence,
        "hard_level": float(1.0 - report.predictability_score) if _is_finite(report.predictability_score) else 0.5,
        "predictability_score": float(report.predictability_score) if _is_finite(report.predictability_score) else 0.5,
        "regime_hint": report.regime_hint,
        "is_intermittent": report.is_intermittent,
        "key_metrics": {
            "hurst": report.hurst,
            "sample_entropy": report.sample_entropy,
            "spectral_entropy": report.spectral_entropy,
            "acf_decay": report.acf_decay,
            "lyapunov": report.lyapunov,
        },
        "alternatives": _get_alternative_families(report),
    }


def _get_alternative_families(report: ForecastabilityReport) -> list[tuple[str, float]]:

    weights = {
        "naive_baseline": 0.0,
        "classical_baseline": 0.0,
        "automl": 0.0,
        "internal_industrial": 0.0,
        "foundation": 0.0,
    }

    if report.regime_hint == "periodic":
        weights["classical_baseline"] += 0.7
        weights["internal_industrial"] += 0.5

    if report.regime_hint == "trend":
        weights["automl"] += 0.6
        weights["classical_baseline"] += 0.4

    if report.regime_hint == "chaotic":
        weights["foundation"] += 0.6
        weights["internal_industrial"] += 0.5

    if report.is_intermittent:
        weights["classical_baseline"] += 0.7

    if _is_finite(report.predictability_score):
        if report.predictability_score > 0.7:
            weights["classical_baseline"] += 0.25
        elif report.predictability_score < 0.15:
            weights["naive_baseline"] += 0.8
        elif report.predictability_score < 0.4:
            weights["foundation"] += 0.3
            weights["internal_industrial"] += 0.25

    ordered = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    return [(name, float(score)) for name, score in ordered[:2] if score > 0.2]
