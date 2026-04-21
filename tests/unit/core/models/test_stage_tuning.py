from fedot_ind.core.models.ts_forecasting.hybrid_ensemble_forecaster import HybridEnsembleForecasterImplementation
from fedot_ind.core.models.ts_forecasting.lagged_ridge_forecaster import LaggedRidgeForecasterImplementation
from fedot_ind.core.models.ts_forecasting.low_rank_lagged_ridge_forecaster import (
    LowRankLaggedRidgeForecasterImplementation,
)
from fedot_ind.core.models.ts_forecasting.okhs_fdmd_forecaster import OKHSFDMDForecasterImplementation
from fedot_ind.core.models.ts_forecasting.progress_policy import ForecastingProgressPolicy
from fedot_ind.core.models.ts_forecasting.stage_tuning import (
    ForecastingStageName,
    build_forecasting_stage_search_spaces,
    build_forecasting_stage_tuning_plan,
)
from fedot_ind.core.models.ts_forecasting.stage_tuning_execution import (
    build_forecasting_stage_tuning_execution,
    run_sequential_stage_tuning,
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


def test_build_stage_tuning_plan_distinguishes_lagged_wrapper_from_ridge_shell():
    lagged_plan = build_forecasting_stage_tuning_plan('lagged_forecaster', {'channel_model': 'ridge'})
    ridge_plan = build_forecasting_stage_tuning_plan('lagged_ridge_forecaster', {'alpha': 1.0})

    assert lagged_plan.canonical_model_name == 'lagged_forecaster'
    assert ridge_plan.canonical_model_name == 'lagged_ridge_forecaster'
    assert 'channel_model' in lagged_plan.groups[-1].parameters
    assert 'alpha' in lagged_plan.groups[-1].parameters
    assert 'alpha' in ridge_plan.groups[-1].parameters


def test_build_stage_search_spaces_for_lagged_wrapper_includes_alpha_and_stride():
    spaces = build_forecasting_stage_search_spaces(
        'lagged_forecaster',
        {'window_size': 16, 'channel_model': 'ridge'},
    )

    assert spaces[0].stage == ForecastingStageName.TRAJECTORY.value
    assert set(spaces[0].parameter_space) == {'window_size', 'stride'}
    assert spaces[1].stage == ForecastingStageName.FORECAST_HEAD.value
    assert set(spaces[1].parameter_space) == {'alpha'}


def test_build_stage_tuning_plan_for_mssa_and_havok_exposes_head_stage():
    mssa_plan = build_forecasting_stage_tuning_plan(
        'mssa_forecaster',
        {'window_size': 16, 'rank': 4, 'head_policy': 'mlp'},
    )
    havok_plan = build_forecasting_stage_tuning_plan(
        'havok_forecaster',
        {'window_size': 16, 'rank': 4, 'head_policy': 'mlp'},
    )

    assert mssa_plan.groups[-1].stage == ForecastingStageName.FORECAST_HEAD.value
    assert 'head_policy' in mssa_plan.groups[-1].parameters
    assert 'head_hidden_dim' in mssa_plan.groups[-1].parameters
    assert havok_plan.groups[-1].stage == ForecastingStageName.FORECAST_HEAD.value
    assert 'forcing_threshold_scale' in havok_plan.groups[-1].parameters
    assert 'head_policy' in havok_plan.groups[-1].parameters


def test_build_stage_search_spaces_for_mssa_and_havok_include_head_parameters():
    mssa_spaces = build_forecasting_stage_search_spaces(
        'mssa_forecaster',
        {'window_size': 16, 'rank': 4, 'head_policy': 'mlp'},
    )
    havok_spaces = build_forecasting_stage_search_spaces(
        'havok_forecaster',
        {'window_size': 16, 'rank': 4, 'head_policy': 'mlp'},
    )

    assert mssa_spaces[-1].stage == ForecastingStageName.FORECAST_HEAD.value
    assert 'head_policy' in mssa_spaces[-1].parameter_space
    assert 'head_epochs' in mssa_spaces[-1].parameter_space
    assert havok_spaces[-1].stage == ForecastingStageName.FORECAST_HEAD.value
    assert 'head_hidden_layers' in havok_spaces[-1].parameter_space
    assert 'forcing_decay' in havok_spaces[-1].parameter_space


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


def test_build_stage_tuning_plan_for_neural_forecasters_marks_them_as_stage_citizens():
    patch_tst_plan = build_forecasting_stage_tuning_plan(
        'patch_tst_model',
        {'patch_len': 16, 'epochs': 20},
    )
    tcn_plan = build_forecasting_stage_tuning_plan(
        'tcn_model',
        {'patch_len': 16, 'kernel_size': 3, 'epochs': 20},
    )
    deepar_plan = build_forecasting_stage_tuning_plan(
        'deepar_model',
        {'hidden_size': 32, 'epochs': 20},
    )
    nbeats_plan = build_forecasting_stage_tuning_plan(
        'nbeats_model',
        {'n_stacks': 2, 'epochs': 20},
    )

    assert patch_tst_plan.family == 'neural_forecaster'
    assert patch_tst_plan.groups[0].stage == ForecastingStageName.TRAJECTORY.value
    assert patch_tst_plan.groups[-1].stage == ForecastingStageName.FORECAST_HEAD.value
    assert patch_tst_plan.metadata['head_runtime'] == 'neural'
    assert tcn_plan.family == 'neural_forecaster'
    assert tcn_plan.groups[0].stage == ForecastingStageName.TRAJECTORY.value
    assert tcn_plan.groups[-1].stage == ForecastingStageName.FORECAST_HEAD.value
    assert deepar_plan.family == 'neural_forecaster'
    assert len(deepar_plan.groups) == 1
    assert deepar_plan.groups[0].stage == ForecastingStageName.FORECAST_HEAD.value
    assert nbeats_plan.family == 'neural_forecaster'
    assert len(nbeats_plan.groups) == 1
    assert nbeats_plan.groups[0].stage == ForecastingStageName.FORECAST_HEAD.value


def test_build_stage_search_spaces_filters_search_space_per_stage():
    spaces = build_forecasting_stage_search_spaces(
        'low_rank_lagged_ridge_forecaster',
        {'window_size': 16, 'stride': 1, 'alpha': 1.0},
    )

    assert spaces[0].stage == ForecastingStageName.TRAJECTORY.value
    assert set(spaces[0].parameter_space) == {'window_size', 'stride'}
    assert spaces[1].stage == ForecastingStageName.DECOMPOSITION_RANK.value
    assert 'decomposition_strategy' in spaces[1].parameter_space
    assert spaces[2].stage == ForecastingStageName.FORECAST_HEAD.value
    assert set(spaces[2].parameter_space) == {'alpha'}


def test_build_stage_search_spaces_supports_neural_forecasters():
    patch_spaces = build_forecasting_stage_search_spaces(
        'patch_tst_model',
        {'patch_len': 16, 'epochs': 20},
    )
    tcn_spaces = build_forecasting_stage_search_spaces(
        'tcn_model',
        {'patch_len': 16, 'epochs': 20, 'kernel_size': 3},
    )
    deepar_spaces = build_forecasting_stage_search_spaces(
        'deepar_model',
        {'hidden_size': 32, 'epochs': 20},
    )
    nbeats_spaces = build_forecasting_stage_search_spaces(
        'nbeats_model',
        {'n_stacks': 2, 'epochs': 20},
    )

    assert patch_spaces[0].stage == ForecastingStageName.TRAJECTORY.value
    assert 'patch_len' in patch_spaces[0].parameter_space
    assert patch_spaces[1].stage == ForecastingStageName.FORECAST_HEAD.value
    assert 'epochs' in patch_spaces[1].parameter_space
    assert tcn_spaces[0].stage == ForecastingStageName.TRAJECTORY.value
    assert 'kernel_size' in tcn_spaces[1].parameter_space
    assert len(deepar_spaces) == 1
    assert 'hidden_size' in deepar_spaces[0].parameter_space
    assert len(nbeats_spaces) == 1
    assert 'n_stacks' in nbeats_spaces[0].parameter_space


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


def test_implementations_publish_stage_search_spaces():
    lagged_spaces = LaggedRidgeForecasterImplementation(
        params={'window_size': 12, 'stride': 1}).get_stage_search_spaces()
    low_rank_spaces = LowRankLaggedRidgeForecasterImplementation(
        params={'window_size': 12, 'stride': 1, 'alpha': 1.0}
    ).get_stage_search_spaces()
    okhs_spaces = OKHSFDMDForecasterImplementation(params={'window_size': 16, 'q': 0.7}).get_stage_search_spaces()
    hybrid_spaces = HybridEnsembleForecasterImplementation(params={'complex_branch': 'havok'}).get_stage_search_spaces()

    assert lagged_spaces[0]['stage'] == ForecastingStageName.TRAJECTORY.value
    assert low_rank_spaces[1]['stage'] == ForecastingStageName.DECOMPOSITION_RANK.value
    assert okhs_spaces[-1]['stage'] == ForecastingStageName.FORECAST_HEAD.value
    assert hybrid_spaces[-1]['stage'] == ForecastingStageName.ENSEMBLE.value


def test_build_stage_tuning_execution_applies_only_stage_owned_parameters():
    execution = build_forecasting_stage_tuning_execution(
        'low_rank_lagged_ridge_forecaster',
        base_params={'window_size': 12, 'stride': 1, 'alpha': 1.0},
        stage_updates={
            ForecastingStageName.TRAJECTORY.value: {'window_size': 24, 'alpha': 5.0},
            ForecastingStageName.DECOMPOSITION_RANK.value: {'decomposition_strategy': 'randomized'},
            ForecastingStageName.FORECAST_HEAD.value: {'alpha': 2.0},
        },
    )

    assert execution.final_parameters['window_size'] == 24
    assert execution.final_parameters['alpha'] == 2.0
    assert execution.steps[0].ignored_parameters == {'alpha': 5.0}
    assert execution.steps[1].applied_parameters == {'decomposition_strategy': 'randomized'}


def test_implementation_publishes_stage_tuning_execution():
    implementation = OKHSFDMDForecasterImplementation(params={'window_size': 16, 'q': 0.7, 'n_modes': 4})
    execution = implementation.get_stage_tuning_execution(
        {
            ForecastingStageName.TRAJECTORY.value: {'window_size': 20},
            ForecastingStageName.FORECAST_HEAD.value: {'n_modes': 6},
        }
    )

    assert execution['canonical_model_name'] == 'okhs_fdmd_forecaster'
    assert execution['final_parameters']['window_size'] == 20
    assert execution['final_parameters']['n_modes'] == 6
    assert execution['steps'][-1]['stage'] == ForecastingStageName.FORECAST_HEAD.value


def test_run_sequential_stage_tuning_optimizes_stage_by_stage():
    def objective(params):
        return abs(params.get('window_size', 0) - 24) + abs(params.get('alpha', 0) - 2.0)

    result = run_sequential_stage_tuning(
        'lagged_ridge_forecaster',
        objective=objective,
        base_params={'window_size': 12, 'stride': 1, 'alpha': 1.0},
        stage_updates={
            ForecastingStageName.TRAJECTORY.value: {'window_size': 24},
            ForecastingStageName.FORECAST_HEAD.value: {'alpha': 2.0},
        },
        max_values_per_parameter=2,
        max_stage_candidates=4,
    )

    assert result.best_parameters['window_size'] == 24
    assert result.best_parameters['alpha'] == 2.0
    assert result.stage_history[0]['stage'] == ForecastingStageName.TRAJECTORY.value
    assert result.stage_history[-1]['stage'] == ForecastingStageName.FORECAST_HEAD.value


def test_run_sequential_stage_tuning_uses_tqdm_progress(monkeypatch):
    calls = []

    def fake_tqdm(iterable=None, *args, **kwargs):
        calls.append(kwargs.get('desc'))
        return iterable

    monkeypatch.setattr(
        'fedot_ind.core.models.ts_forecasting.stage_tuning_execution.tqdm',
        fake_tqdm,
    )

    run_sequential_stage_tuning(
        'lagged_ridge_forecaster',
        objective=lambda params: abs(params.get('window_size', 0) - 16) + abs(params.get('alpha', 0) - 1.0),
        base_params={'window_size': 12, 'stride': 1, 'alpha': 2.0},
        stage_updates={
            ForecastingStageName.TRAJECTORY.value: {'window_size': 16},
            ForecastingStageName.FORECAST_HEAD.value: {'alpha': 1.0},
        },
        max_values_per_parameter=2,
        max_stage_candidates=4,
        show_progress=True,
    )

    assert calls
    assert any(desc and 'Stage tuning:' in desc for desc in calls)


def test_run_sequential_stage_tuning_preserves_progress_policy_metadata():
    result = run_sequential_stage_tuning(
        'lagged_ridge_forecaster',
        objective=lambda params: abs(params.get('window_size', 0) - 16) + abs(params.get('alpha', 0) - 1.0),
        base_params={'window_size': 12, 'stride': 1, 'alpha': 2.0},
        stage_updates={
            ForecastingStageName.TRAJECTORY.value: {'window_size': 16},
            ForecastingStageName.FORECAST_HEAD.value: {'alpha': 1.0},
        },
        max_values_per_parameter=2,
        max_stage_candidates=4,
        progress_policy=ForecastingProgressPolicy(enabled=True, stage_tuning_enabled=True),
    )

    assert result.metadata['progress_policy']['enabled'] is True
    assert result.metadata['progress_policy']['stage_tuning_enabled'] is True


def test_implementation_publishes_run_stage_tuning_result():
    implementation = LaggedRidgeForecasterImplementation(params={'window_size': 12, 'stride': 1, 'alpha': 1.0})

    result = implementation.run_stage_tuning(
        objective=lambda params: abs(params.get('window_size', 0) - 20) + abs(params.get('alpha', 0) - 2.0),
        stage_updates={
            ForecastingStageName.TRAJECTORY.value: {'window_size': 20},
            ForecastingStageName.FORECAST_HEAD.value: {'alpha': 2.0},
        },
    )

    assert result['best_parameters']['window_size'] == 20
    assert result['best_parameters']['alpha'] == 2.0
    assert result['stage_history'][-1]['stage'] == ForecastingStageName.FORECAST_HEAD.value
