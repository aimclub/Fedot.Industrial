from .regime_diagnostics import RegimeDiagnosticsResult, analyze_regime_diagnostics
from .regime_routing import adapter_name_to_family, RegimeRoutingDecision, RegimeRoutingPolicy, \
    recommend_forecasting_model
from .stage_tuning import (
    ForecastingStageName,
    ForecastingStageTuningPlan,
    StageTuningGroup,
    build_forecasting_stage_tuning_plan,
)

try:  # pragma: no cover - tensor-native forecasting stack requires torch
    from .forecasting_runtime import (
        DecompositionResult,
        ForecastHeadResult,
        ForecastTensorBatch,
        ForecastingBoundaryAdapter,
        ForecastingEvaluationResult,
        ForecastingOperationCapability,
        ForecastingRuntimeAdapter,
        ForecastingSplitSpec,
        RankTruncationResult,
        TensorDevicePolicy,
        TrajectoryTransformResult,
    )
    from .hybrid_ensemble_forecaster import HybridEnsembleForecaster, HybridEnsembleForecasterImplementation
    from .lagged_ridge_forecaster import LaggedRidgeForecaster, LaggedRidgeForecasterImplementation
    from .low_rank_lagged_ridge_forecaster import (
        LowRankLaggedRidgeForecaster,
        LowRankLaggedRidgeForecasterImplementation,
    )
    from .okhs_fdmd_forecaster import OKHSFDMDForecaster, OKHSFDMDForecasterImplementation
except Exception:  # pragma: no cover - keep regime-level utilities importable in lightweight envs
    pass

__all__ = [
    'RegimeDiagnosticsResult',
    'RegimeRoutingDecision',
    'RegimeRoutingPolicy',
    'adapter_name_to_family',
    'analyze_regime_diagnostics',
    'recommend_forecasting_model',
    'ForecastingStageName',
    'ForecastingStageTuningPlan',
    'StageTuningGroup',
    'build_forecasting_stage_tuning_plan',
]

for _optional_symbol in (
        'DecompositionResult',
        'ForecastHeadResult',
        'ForecastTensorBatch',
        'ForecastingBoundaryAdapter',
        'ForecastingEvaluationResult',
        'ForecastingOperationCapability',
        'ForecastingRuntimeAdapter',
        'ForecastingSplitSpec',
        'RankTruncationResult',
        'TensorDevicePolicy',
        'TrajectoryTransformResult',
        'LaggedRidgeForecaster',
        'LaggedRidgeForecasterImplementation',
        'LowRankLaggedRidgeForecaster',
        'LowRankLaggedRidgeForecasterImplementation',
        'HybridEnsembleForecaster',
        'HybridEnsembleForecasterImplementation',
        'OKHSFDMDForecaster',
        'OKHSFDMDForecasterImplementation',
):
    if _optional_symbol in globals():
        __all__.append(_optional_symbol)
