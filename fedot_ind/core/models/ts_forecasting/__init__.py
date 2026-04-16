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
    from .neural_forecast_head import (
        DeepARForecastHeadImplementation,
        NEURAL_FORECASTING_MODEL_REGISTRY,
        NBeatsForecastHeadImplementation,
        NeuralForecastHead,
        NeuralForecastHeadImplementation,
        NeuralForecastHeadRunResult,
        NeuralForecastHeadSpec,
        PatchTSTForecastHeadImplementation,
        TCNForecastHeadImplementation,
        build_neural_forecast_head,
        build_neural_forecasting_input_data,
        build_neural_forecasting_stage_diagnostics,
        normalize_neural_forecast_prediction,
        run_neural_forecast_head_on_series,
        resolve_neural_forecasting_model_cls,
    )
    from .neural_forecast_head_bridge import (
        NeuralForecastHeadBridge,
    )
    from .okhs_fdmd_forecaster import OKHSFDMDForecaster, OKHSFDMDForecasterImplementation
    from .okhs_fdmd_forecaster import (
        OKHSFDMDForecasterRunResult,
        OKHSFDMDForecasterSpec,
        build_okhs_fdmd_forecaster,
        build_okhs_fdmd_spec,
        build_okhs_fdmd_runtime_diagnostics,
        normalize_okhs_fdmd_params,
        normalize_okhs_fdmd_prediction,
        run_okhs_fdmd_forecaster_on_series,
    )
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
        'DeepARForecastHeadImplementation',
        'NBeatsForecastHeadImplementation',
        'NeuralForecastHead',
        'NeuralForecastHeadImplementation',
        'NeuralForecastHeadRunResult',
        'NeuralForecastHeadSpec',
        'NeuralForecastHeadBridge',
        'NEURAL_FORECASTING_MODEL_REGISTRY',
        'PatchTSTForecastHeadImplementation',
        'TCNForecastHeadImplementation',
        'build_neural_forecast_head',
        'build_neural_forecasting_input_data',
        'build_neural_forecasting_stage_diagnostics',
        'normalize_neural_forecast_prediction',
        'run_neural_forecast_head_on_series',
        'resolve_neural_forecasting_model_cls',
        'OKHSFDMDForecaster',
        'OKHSFDMDForecasterImplementation',
        'OKHSFDMDForecasterRunResult',
        'OKHSFDMDForecasterSpec',
        'build_okhs_fdmd_forecaster',
        'build_okhs_fdmd_spec',
        'build_okhs_fdmd_runtime_diagnostics',
        'normalize_okhs_fdmd_params',
        'normalize_okhs_fdmd_prediction',
        'run_okhs_fdmd_forecaster_on_series',
):
    if _optional_symbol in globals():
        __all__.append(_optional_symbol)
