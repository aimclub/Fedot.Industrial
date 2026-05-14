from benchmark.v2 import (
    ArtifactSpec,
    BenchmarkSuiteConfig,
    DatasetSpec,
    ModelSpec,
    RunSpec,
    TaskType,
    run_forecasting_benchmark_suite,
)

EXPERIMENT_DATE = '220426'
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
    # ModelSpec(
    #     adapter_name='lagged_forecaster',
    #     display_name='lagged_forecaster',
    #     params={
    #         'window_size': 10,
    #         'channel_model': 'ridge',
    #     },
    # ),
    ModelSpec(
        adapter_name='deepar_model',
        display_name='deepar_model',
        params={
            'epochs': 20,
            'batch_size': 16,
            'learning_rate': 1e-3,
            'cell_type': 'LSTM',
            'rnn_layers': 2,
            'hidden_size': 32,
            'expected_distribution': 'normal',
            'dropout': 0.1,
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
    # ModelSpec(
    #     adapter_name='ssa_forecaster',
    #     display_name='ssa_forecaster',
    #     params={
    #         'window_size': None,
    #         'rank': None,
    #         'explained_variance': 0.95,
    #     },
    # ),
    # ModelSpec(
    #     adapter_name='okhs',
    #     display_name='okhs_forecasting',
    #     params={
    #         'method': 'dmd',
    #         'q': 0.9,
    #         'window_size': 8,
    #         'window_policy': 'adaptive_cycle_aware',
    #         'trajectory_representation_policy': 'projected',
    #         'latent_trajectory_stride_policy': 'adaptive',
    #         'mode_selection_policy': 'energy',
    #         'mode_energy_threshold': 0.95,
    #         'prediction_mode_selection_policy': 'adaptive_tail_energy',
    #         'boundary_alignment_policy': 'tapered_offset',
    #         'anti_smoothing_policy': 'residual_bridge',
    #         'n_modes': 2,
    #     },
    # ),
)

config = BenchmarkSuiteConfig(
    task_type=TaskType.FORECASTING,
    datasets=M4_DATASETS,
    models=FORECASTING_MODELS,
    metrics=('mase', 'smape', 'owa', 'rmse', 'mae'),
    artifact_spec=ArtifactSpec(
        output_dir=f'benchmark/results/v2_demo/m4_regime_suite_{EXPERIMENT_DATE}',
        persist_on_run=True,
    ),
    run_spec=RunSpec(
        run_name='m4_regime_suite',
        primary_metric='mae',
        show_progress=True,
        progress_leave=False,
        progress_log_errors=True,
        progress_log_summaries=True,
    ),
)

result = run_forecasting_benchmark_suite(config)
