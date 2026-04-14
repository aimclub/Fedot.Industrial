from fedot_ind.core.models.ts_forecasting.hybrid_ensemble_forecaster import HybridEnsembleForecasterImplementation
from fedot_ind.core.models.ts_forecasting.lagged_ridge_forecaster import LaggedRidgeForecasterImplementation
from fedot_ind.core.models.ts_forecasting.low_rank_lagged_ridge_forecaster import (
    LowRankLaggedRidgeForecasterImplementation,
)
from fedot_ind.core.models.ts_forecasting.okhs_fdmd_forecaster import OKHSFDMDForecasterImplementation
from fedot_ind.core.models.ts_forecasting.stage_tuning import (
    ForecastingStageName,
    build_forecasting_stage_tuning_plan,
)


def test_build_stage_tuning_plan_for_low_rank_forecaster_returns_ordered_groups():
    plan = build_forecasting_stage_tuning_plan(
        'low_rank_lagged_ridge_forecaster',
        {
            'window_size': 16,
            'stride': 2,
            'alpha': 1.0,
            'decomposition_strategy': 'randomized',
        },
    )

    stages = tuple(group.stage for group in plan.groups)
    assert plan.family == 'low_rank_linear'
    assert stages == (
        ForecastingStageName.TRAJECTORY.value,
        ForecastingStageName.DECOMPOSITION_RANK.value,
        ForecastingStageName.FORECAST_HEAD.value,
    )
    assert 'alpha' in plan.groups[-1].parameters


def test_build_stage_tuning_plan_for_okhs_forecaster_separates_head_from_representation():
    plan = build_forecasting_stage_tuning_plan(
        'okhs_fdmd_forecaster',
        {'trajectory_representation_policy': 'projected', 'q': 0.7, 'n_modes': 4},
    )

    assert plan.family == 'operator_model'
    assert plan.groups[1].stage == ForecastingStageName.DECOMPOSITION_RANK.value
    assert 'trajectory_representation_policy' in plan.groups[1].parameters
    assert 'n_modes' in plan.groups[2].parameters
    assert 'boundary_alignment_policy' in plan.groups[2].parameters


def test_build_stage_tuning_plan_for_hybrid_ensemble_exposes_branch_and_ensemble_groups():
    plan = build_forecasting_stage_tuning_plan(
        'hybrid_ensemble_forecaster',
        {'complex_branch': 'havok'},
    )

    assert plan.family == 'hybrid_ensemble'
    assert plan.groups[0].stage == ForecastingStageName.TRAJECTORY.value
    assert plan.groups[1].stage == ForecastingStageName.ENSEMBLE.value
    assert plan.groups[0].metadata['branch_models'][-1] == 'havok'


def test_implementations_publish_stage_tuning_plans():
    lagged_plan = LaggedRidgeForecasterImplementation(params={'window_size': 12, 'stride': 1}).get_stage_tuning_plan()
    low_rank_plan = LowRankLaggedRidgeForecasterImplementation(
        params={'window_size': 12, 'stride': 1, 'alpha': 1.0}
    ).get_stage_tuning_plan()
    okhs_plan = OKHSFDMDForecasterImplementation(params={'window_size': 16, 'q': 0.7}).get_stage_tuning_plan()
    hybrid_plan = HybridEnsembleForecasterImplementation(params={'complex_branch': 'okhs'}).get_stage_tuning_plan()

    assert lagged_plan['canonical_model_name'] == 'lagged_ridge_forecaster'
    assert low_rank_plan['canonical_model_name'] == 'low_rank_lagged_ridge_forecaster'
    assert okhs_plan['canonical_model_name'] == 'okhs_fdmd_forecaster'
    assert hybrid_plan['canonical_model_name'] == 'hybrid_ensemble_forecaster'
