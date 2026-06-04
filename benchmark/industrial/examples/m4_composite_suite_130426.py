from benchmark.industrial import (
    ArtifactSpec,
    BenchmarkSuiteConfig,
    DatasetSpec,
    ModelSpec,
    RunSpec,
    TaskType,
    run_forecasting_benchmark_suite,
)

EXPERIMENT_DATE = '130426'
M4_SUBSETS = ('daily', 'weekly', 'monthly', 'quarterly', 'yearly')

M4_DATASETS = tuple(
    DatasetSpec(
        benchmark='m4',
        dataset_name=f'm4_{subset.lower()}_full',
        subset=subset,
        sample_size=None,
        adapter_options={'use_local_files': True},
    )
    for subset in M4_SUBSETS
)

FORECASTING_MODELS = (
    ModelSpec(
        adapter_name='lagged_ridge_forecaster',
        display_name='lagged_ridge_forecaster',
        params={
            'window_size': 12,
            'stride': 1,
            'alpha': 1.0,
        },
    ),
    ModelSpec(
        adapter_name='low_rank_lagged_ridge_forecaster',
        display_name='low_rank_lagged_ridge_forecaster',
        params={
            'window_size': 12,
            'stride': 1,
            'alpha': 1.0,
            'explained_variance': 0.95,
            'decomposition_strategy': 'full',
            'rank_truncation_policy': 'explained_variance',
        },
    ),
    ModelSpec(
        adapter_name='hybrid_ensemble_forecaster',
        display_name='hybrid_ensemble_forecaster',
        params={
            'complex_branch': 'havok',
            'lagged_params': {'window_size': 12, 'stride': 1, 'alpha': 1.0},
            'low_rank_params': {'window_size': 12, 'stride': 1, 'explained_variance': 0.95},
            'complex_params': {'window_size': 14, 'rank': 4, 'forcing_threshold_scale': 0.85},
        },
    ),
    ModelSpec(
        adapter_name='havok',
        display_name='havok_forecaster',
        params={
            'window_size': None,
            'rank': 4,
            'forcing_threshold_scale': 0.85,
            'forcing_decay': 0.85,
        },
    ),
    ModelSpec(
        adapter_name='okhs',
        display_name='okhs_forecasting',
        params={
            'method': 'dmd',
            'q': 0.9,
            'window_size': 8,
            'window_policy': 'adaptive_cycle_aware',
            'trajectory_representation_policy': 'projected',
            'latent_trajectory_stride_policy': 'adaptive',
            'mode_selection_policy': 'energy',
            'mode_energy_threshold': 0.95,
            'prediction_mode_selection_policy': 'adaptive_tail_energy',
            'boundary_alignment_policy': 'tapered_offset',
            'anti_smoothing_policy': 'residual_bridge',
            'n_modes': 2,
        },
    ),
)

config = BenchmarkSuiteConfig(
    task_type=TaskType.FORECASTING,
    datasets=M4_DATASETS,
    models=FORECASTING_MODELS,
    metrics=('mase', 'smape', 'owa', 'rmse', 'mae'),
    artifact_spec=ArtifactSpec(
        output_dir=f'benchmark/results/industrial_demo/m4_composite_suite_{EXPERIMENT_DATE}',
        persist_on_run=True,
    ),
    run_spec=RunSpec(
        run_name='m4_composite_suite',
        primary_metric='mae',
        show_progress=True,
        progress_leave=False,
        progress_log_errors=True,
        progress_log_summaries=True,
    ),
)

result = run_forecasting_benchmark_suite(config)
