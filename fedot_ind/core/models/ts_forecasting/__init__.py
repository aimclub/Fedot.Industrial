from fedot_ind.core.models.ts_forecasting.forecast_tuning.stage_tuning import (
    ForecastingStageSearchSpace,
    ForecastingStageName,
    ForecastingStageTuningPlan,
    StageTuningGroup,
    build_forecasting_stage_search_spaces,
    build_forecasting_stage_tuning_plan,
)
from fedot_ind.core.models.ts_forecasting.forecast_tuning.stage_tuning_execution import (
    ForecastingSequentialStageTuningResult,
    ForecastingStageTuningExecution,
    StageTuningExecutionStep,
    build_forecasting_stage_tuning_execution,
    run_sequential_stage_tuning,
)
from fedot_ind.core.models.ts_forecasting.regime_utils.regime_diagnostics import (
    RegimeDiagnosticsResult,
    analyze_regime_diagnostics,
)
from fedot_ind.core.models.ts_forecasting.regime_utils.regime_routing import (
    RegimeRoutingDecision,
    RegimeRoutingPolicy,
    adapter_name_to_family,
    recommend_forecasting_model,
)
from .progress_policy import ForecastingProgressPolicy, resolve_forecasting_progress_policy

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
    from fedot_ind.core.models.ts_forecasting.ensemble_models.hybrid_ensemble_forecaster import \
        HybridEnsembleForecaster, HybridEnsembleForecasterImplementation
    from fedot_ind.core.models.ts_forecasting.lagged_model.lagged_ridge_forecaster import LaggedRidgeForecaster, \
        LaggedRidgeForecasterImplementation
    from fedot_ind.core.models.ts_forecasting.lagged_model.low_rank_lagged_ridge_forecaster import (
        LowRankLaggedRidgeForecaster,
        LowRankLaggedRidgeForecasterImplementation,
    )
    from fedot_ind.core.models.ts_forecasting.lagged_model.mssa_forecaster import (
        MSSAForecaster,
        MSSAForecasterImplementation,
    )
    from fedot_ind.core.models.ts_forecasting.lagged_model.ssa_forecaster import SSAForecasterImplementation
    from fedot_ind.core.models.ts_forecasting.lagged_model.topo_forecaster import (
        TopologicalAR,
        TopologicalRidgeForecaster,
    )
    from fedot_ind.core.models.ts_forecasting.neural_models.neural_forecast_head import (
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
    from fedot_ind.core.models.ts_forecasting.neural_models.neural_forecast_head_bridge import (
        NeuralForecastHeadBridge,
    )
    from fedot_ind.core.models.ts_forecasting.dmd_models.okhs_fdmd_forecaster import OKHSFDMDForecaster, \
        OKHSFDMDForecasterImplementation
    from fedot_ind.core.models.ts_forecasting.dmd_models.okhs_fdmd_forecaster import (
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
    'StageTuningGroup',
    'StageTuningExecutionStep',
    'build_forecasting_stage_search_spaces',
    'build_forecasting_stage_tuning_plan',
    'build_forecasting_stage_tuning_execution',
    'run_sequential_stage_tuning',
    'ForecastingProgressPolicy',
    'resolve_forecasting_progress_policy',
]

try:  # pragma: no cover - forecasting runtime depends on torch-heavy stack
    from fedot_ind.core.models.ts_forecasting.forecast_tuning.stage_tuning_runtime import (
        ForecastingSeriesEvaluation,
        ForecastingSeriesStageTuningResult,
        build_forecasting_stage_objective_from_series,
        evaluate_forecasting_model_on_series,
        run_forecasting_stage_tuning_on_series,
    )

    __all__.extend([
        'ForecastingSeriesEvaluation',
        'ForecastingSeriesStageTuningResult',
        'build_forecasting_stage_objective_from_series',
        'evaluate_forecasting_model_on_series',
        'run_forecasting_stage_tuning_on_series',
    ])
except Exception:  # pragma: no cover
    pass

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
        'MSSAForecaster',
        'MSSAForecasterImplementation',
        'SSAForecasterImplementation',
        'TopologicalAR',
        'TopologicalRidgeForecaster',
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
