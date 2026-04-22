from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from fedot_ind.core.repository.forecasting_registry import canonical_forecasting_model_name
from .regime_diagnostics import RegimeDiagnosticsResult


class RoutingAdapterName(str, Enum):
    MSSA = 'mssa'
    SSA_COMPAT = 'ssa_compat'
    HAVOK = 'havok'
    OKHS = 'okhs'
    OKHS_FDMD = 'okhs_fdmd_forecaster'
    LAGGED_RIDGE = 'lagged_ridge_forecaster'
    LOW_RANK_LAGGED_RIDGE = 'low_rank_lagged_ridge_forecaster'
    HYBRID_ENSEMBLE = 'hybrid_ensemble_forecaster'
    LINEAR_TREND = 'linear_trend'
    NAIVE_LAST_VALUE = 'naive_last_value'


def adapter_name_to_family(adapter_name: str) -> str:
    normalized = canonical_forecasting_model_name(adapter_name)
    if normalized in {
        RoutingAdapterName.LAGGED_RIDGE.value,
        'lagged_forecaster',
        'topo_forecaster',
        'ridge_forecasting_head',
    }:
        return 'lagged_linear'
    if normalized in {
        RoutingAdapterName.LOW_RANK_LAGGED_RIDGE.value,
        RoutingAdapterName.MSSA.value,
        'mssa_forecaster',
        RoutingAdapterName.SSA_COMPAT.value,
        'ssa_forecaster',
    }:
        return 'low_rank_linear'
    if normalized in {
        RoutingAdapterName.OKHS.value,
        RoutingAdapterName.OKHS_FDMD.value,
        RoutingAdapterName.HAVOK.value,
        'havok_forecaster',
        RoutingAdapterName.HYBRID_ENSEMBLE.value,
        'classical_dmd',
    }:
        return 'operator_model'
    if normalized in {'deepar_model', 'nbeats_model', 'patch_tst_model', 'tcn_model', 'tft'}:
        return 'neural_forecaster'
    return 'simple_baseline'


@dataclass(frozen=True)
class RegimeRoutingPolicy:
    periodic_concentration_min: float = 0.25
    periodic_flatness_max: float = 0.25
    periodic_acf_min: float = 0.5
    switching_score_min: float = 0.4
    local_linearity_min: float = 0.55
    short_history_length: int = 24
    strong_period_length: int = 6


@dataclass(frozen=True)
class RegimeRoutingDecision:
    regime_hint: str
    primary_adapter: str
    candidate_adapters: tuple[str, ...]
    fallback_adapter: str
    confidence: float
    rationale: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def recommend_forecasting_model(
        diagnostics: RegimeDiagnosticsResult,
        policy: RegimeRoutingPolicy | None = None,
) -> RegimeRoutingDecision:
    resolved = policy or RegimeRoutingPolicy()
    regime_hint = diagnostics.regime_hint
    rationale: list[str] = [f'regime_hint={regime_hint}']

    if regime_hint == 'insufficient_history':
        rationale.append(f'series_length={diagnostics.series_length} is below stable modeling range')
        return RegimeRoutingDecision(
            regime_hint=regime_hint,
            primary_adapter=RoutingAdapterName.NAIVE_LAST_VALUE.value,
            candidate_adapters=(
                RoutingAdapterName.NAIVE_LAST_VALUE.value,
                RoutingAdapterName.LINEAR_TREND.value,
            ),
            fallback_adapter=RoutingAdapterName.NAIVE_LAST_VALUE.value,
            confidence=0.2,
            rationale=tuple(rationale),
        )

    periodic_signature_detected = (
            diagnostics.acf_decay_rate >= resolved.periodic_acf_min
            and diagnostics.spectral_flatness <= resolved.periodic_flatness_max
            and diagnostics.switching_score < resolved.switching_score_min
    )

    if regime_hint == 'periodic' or (
            diagnostics.dominant_period is not None
            and diagnostics.spectral_concentration >= resolved.periodic_concentration_min
    ) or periodic_signature_detected:
        adapter = (
            RoutingAdapterName.MSSA
            if (diagnostics.dominant_period or 0) >= resolved.strong_period_length
            else RoutingAdapterName.SSA_COMPAT
        )
        rationale.append(f'dominant_period={diagnostics.dominant_period}')
        rationale.append(f'spectral_concentration={diagnostics.spectral_concentration:.3f}')
        return RegimeRoutingDecision(
            regime_hint='periodic',
            primary_adapter=adapter.value,
            candidate_adapters=(adapter.value, RoutingAdapterName.SSA_COMPAT.value, RoutingAdapterName.MSSA.value),
            fallback_adapter=RoutingAdapterName.LINEAR_TREND.value,
            confidence=min(0.98, 0.45 + diagnostics.spectral_concentration + 0.2 * diagnostics.acf_decay_rate),
            rationale=tuple(dict.fromkeys(rationale)),
        )

    switching_signature_detected = diagnostics.switching_score >= resolved.switching_score_min

    if regime_hint == 'switching' or switching_signature_detected:
        rationale.append(f'switching_score={diagnostics.switching_score:.3f}')
        rationale.append(f'local_linearity_score={diagnostics.local_linearity_score:.3f}')
        return RegimeRoutingDecision(
            regime_hint='switching',
            primary_adapter=RoutingAdapterName.HAVOK.value,
            candidate_adapters=(RoutingAdapterName.HAVOK.value, RoutingAdapterName.LINEAR_TREND.value),
            fallback_adapter=RoutingAdapterName.LINEAR_TREND.value,
            confidence=min(0.97, 0.4 + diagnostics.switching_score),
            rationale=tuple(rationale),
        )

    if regime_hint == 'locally_linear' or diagnostics.local_linearity_score >= resolved.local_linearity_min:
        rationale.append(f'local_linearity_score={diagnostics.local_linearity_score:.3f}')
        rationale.append(f'spectral_flatness={diagnostics.spectral_flatness:.3f}')
        return RegimeRoutingDecision(
            regime_hint='locally_linear',
            primary_adapter=RoutingAdapterName.OKHS.value,
            candidate_adapters=(RoutingAdapterName.OKHS.value, RoutingAdapterName.LINEAR_TREND.value),
            fallback_adapter=RoutingAdapterName.LINEAR_TREND.value,
            confidence=min(0.95, 0.35 + diagnostics.local_linearity_score),
            rationale=tuple(rationale),
        )

    rationale.append(f'spectral_flatness={diagnostics.spectral_flatness:.3f}')
    rationale.append('weak_structure fallback to simple deterministic baseline')
    fallback = (
        RoutingAdapterName.NAIVE_LAST_VALUE
        if diagnostics.series_length <= resolved.short_history_length
        else RoutingAdapterName.LINEAR_TREND
    )
    return RegimeRoutingDecision(
        regime_hint='weak_structure',
        primary_adapter=fallback.value,
        candidate_adapters=(fallback.value, RoutingAdapterName.NAIVE_LAST_VALUE.value),
        fallback_adapter=RoutingAdapterName.NAIVE_LAST_VALUE.value,
        confidence=max(0.2, 0.45 - 0.25 * diagnostics.spectral_concentration),
        rationale=tuple(rationale),
    )
