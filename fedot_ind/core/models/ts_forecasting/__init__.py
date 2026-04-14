from .regime_diagnostics import RegimeDiagnosticsResult, analyze_regime_diagnostics
from .regime_routing import adapter_name_to_family, RegimeRoutingDecision, RegimeRoutingPolicy, \
    recommend_forecasting_model
from .stage_tuning import (
    ForecastingStageSearchSpace,
    ForecastingStageName,
    ForecastingStageTuningPlan,
    StageTuningGroup,
    build_forecasting_stage_search_spaces,
    build_forecasting_stage_tuning_plan,
)
from .stage_tuning_execution import (
    ForecastingSequentialStageTuningResult,
    ForecastingStageTuningExecution,
    StageTuningExecutionStep,
    build_forecasting_stage_tuning_execution,
    run_sequential_stage_tuning,
)
from .stage_tuning_runtime import (
    ForecastingSeriesEvaluation,
    ForecastingSeriesStageTuningResult,
    build_forecasting_stage_objective_from_series,
    evaluate_forecasting_model_on_series,
    run_forecasting_stage_tuning_on_series,
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
    from .neural_forecast_head_bridge import (
        NEURAL_FORECASTING_MODEL_REGISTRY,
        NeuralForecastHeadBridge,
        build_neural_forecasting_stage_diagnostics,
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
    'ForecastingStageSearchSpace',
    'ForecastingStageTuningPlan',
    'ForecastingStageTuningExecution',
    'ForecastingSequentialStageTuningResult',
    'ForecastingSeriesEvaluation',
    'ForecastingSeriesStageTuningResult',
    'StageTuningGroup',
    'StageTuningExecutionStep',
    'build_forecasting_stage_search_spaces',
    'build_forecasting_stage_objective_from_series',
    'build_forecasting_stage_tuning_plan',
    'build_forecasting_stage_tuning_execution',
    'evaluate_forecasting_model_on_series',
    'run_forecasting_stage_tuning_on_series',
    'run_sequential_stage_tuning',
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
        'NeuralForecastHeadBridge',
        'NEURAL_FORECASTING_MODEL_REGISTRY',
        'build_neural_forecasting_stage_diagnostics',
        'OKHSFDMDForecaster',
        'OKHSFDMDForecasterImplementation',
):
    if _optional_symbol in globals():
        __all__.append(_optional_symbol)
