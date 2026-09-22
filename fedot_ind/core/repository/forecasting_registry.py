from __future__ import annotations

from fedot_ind.core.repository.runtime_operation_registry import LazyOperationMapping

FORECASTING_MODEL_ALIASES: dict[str, str] = {
    'mssa': 'mssa_forecaster',
    'havok': 'havok_forecaster',
}

CANONICAL_STAGE_FORECASTING_MODELS: tuple[str, ...] = (
    'lagged_ridge_forecaster',
    'topo_forecaster',
    'low_rank_lagged_ridge_forecaster',
    'ssa_forecaster',
    'mssa_forecaster',
    'havok_forecaster',
    'okhs_fdmd_forecaster',
    'hybrid_ensemble_forecaster',
)

FORECASTING_MODEL_TARGETS: dict[str, str] = {
    'eigen_forecaster': 'fedot_ind.core.models.ts_forecasting.lagged_strategy.eigen_forecaster:EigenAR',
    'topo_forecaster': 'fedot_ind.core.models.ts_forecasting.lagged_model.topo_forecaster:TopologicalAR',
    'lagged_forecaster': 'fedot_ind.core.models.ts_forecasting.lagged_strategy.lagged_forecaster:LaggedAR',
    'lagged_ridge_forecaster': (
        'fedot_ind.core.models.ts_forecasting.lagged_model.lagged_ridge_forecaster:'
        'LaggedRidgeForecasterImplementation'
    ),
    'low_rank_lagged_ridge_forecaster': (
        'fedot_ind.core.models.ts_forecasting.lagged_model.low_rank_lagged_ridge_forecaster:'
        'LowRankLaggedRidgeForecasterImplementation'
    ),
    'hybrid_ensemble_forecaster': (
        'fedot_ind.core.models.ts_forecasting.ensemble_models.hybrid_ensemble_forecaster:'
        'HybridEnsembleForecasterImplementation'
    ),
    'okhs_fdmd_forecaster': (
        'fedot_ind.core.models.ts_forecasting.dmd_models.okhs_fdmd_forecaster:'
        'OKHSFDMDForecasterImplementation'
    ),
    'ssa_forecaster': 'fedot_ind.core.models.ts_forecasting.lagged_model.ssa_forecaster:SSAForecasterImplementation',
    'mssa_forecaster': 'fedot_ind.core.models.ts_forecasting.lagged_model.mssa_forecaster:MSSAForecasterImplementation',
    'havok_forecaster': 'fedot_ind.core.models.ts_forecasting.dmd_models.havok_forecaster:HAVOKForecasterImplementation',
    'patch_tst_model': (
        'fedot_ind.core.models.ts_forecasting.neural_models.neural_forecast_head:'
        'PatchTSTForecastHeadImplementation'
    ),
    'tst_model': 'fedot_ind.core.models.ts_forecasting.neural_models.neural_forecast_head:TSTForecastHeadImplementation',
    'deepar_model': (
        'fedot_ind.core.models.ts_forecasting.neural_models.neural_forecast_head:'
        'DeepARForecastHeadImplementation'
    ),
    'tcn_model': 'fedot_ind.core.models.ts_forecasting.neural_models.neural_forecast_head:TCNForecastHeadImplementation',
    'nbeats_model': (
        'fedot_ind.core.models.ts_forecasting.neural_models.neural_forecast_head:'
        'NBeatsForecastHeadImplementation'
    ),
    'glm': 'fedot_ind.core.models.ts_forecasting.glm:GLMIndustrial',
}

FORECASTING_PREPROCESSING_TARGETS: dict[str, str] = {
    'hankelisation': (
        'fedot_ind.core.operation.transformation.data.hankelisation:'
        'HankelisationImplementation'
    ),
    'svd_decomposition': (
        'fedot_ind.core.operation.transformation.data.forecasting_primitives:'
        'SVDDecompositionImplementation'
    ),
    'randomized_svd_decomposition': (
        'fedot_ind.core.operation.transformation.data.forecasting_primitives:'
        'RandomizedSVDDecompositionImplementation'
    ),
    'tensor_decomposition': (
        'fedot_ind.core.operation.transformation.data.forecasting_primitives:'
        'TensorDecompositionImplementation'
    ),
    'explained_variance_truncation': (
        'fedot_ind.core.operation.transformation.data.forecasting_primitives:'
        'ExplainedVarianceRankTruncationImplementation'
    ),
    'statistical_rank_truncation': (
        'fedot_ind.core.operation.transformation.data.forecasting_primitives:'
        'StatisticalRankTruncationImplementation'
    ),
    'expert_rank_truncation': (
        'fedot_ind.core.operation.transformation.data.forecasting_primitives:'
        'ExpertRankTruncationImplementation'
    ),
}

LEGACY_FEDOT_FORECASTING_MODELS = ('ar', 'stl_arima', 'ets')
LEGACY_FEDOT_FORECASTING_PREPROCESSING = ('lagged', 'smoothing', 'gaussian_filter', 'exog_ts')


def _legacy_forecasting_model(name: str) -> type:
    from fedot_ind.core.repository.model_repository import FORECASTING_MODELS

    return FORECASTING_MODELS[name]


def _legacy_forecasting_preprocessing(name: str) -> type:
    from fedot_ind.core.repository.model_repository import FORECASTING_PREPROC

    return FORECASTING_PREPROC[name]


FORECASTING_MODELS = LazyOperationMapping(
    FORECASTING_MODEL_TARGETS,
    fallback=_legacy_forecasting_model,
    extra_names=LEGACY_FEDOT_FORECASTING_MODELS,
)
FORECASTING_PREPROCESSING = LazyOperationMapping(
    FORECASTING_PREPROCESSING_TARGETS,
    fallback=_legacy_forecasting_preprocessing,
    extra_names=LEGACY_FEDOT_FORECASTING_PREPROCESSING,
)


def canonical_forecasting_model_name(name: str | None) -> str:
    normalized = str(name or '').strip().lower()
    return FORECASTING_MODEL_ALIASES.get(normalized, normalized)


def forecasting_aliases_for(model_name: str) -> tuple[str, ...]:
    canonical = canonical_forecasting_model_name(model_name)
    aliases = [alias for alias, target in FORECASTING_MODEL_ALIASES.items() if target == canonical]
    return tuple(dict.fromkeys([canonical, *sorted(aliases)]))
