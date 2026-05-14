from benchmark.v2 import (
    ArtifactSpec,
    BenchmarkSuiteConfig,
    DatasetSpec,
    ModelSpec,
    RunSpec,
    TaskType,
    run_forecasting_benchmark_suite,
)

EXPERIMENT_DATE = '140426'
M4_SUBSETS = ('monthly', 'quarterly')

M4_DATASETS = tuple(
    DatasetSpec(
        benchmark='m4',
        dataset_name=f'm4_{subset.lower()}_neural_stage',
        subset=subset,
        sample_size=50,
        random_seed=42,
        adapter_options={'use_local_files': True},
    )
    for subset in M4_SUBSETS
)

FORECASTING_MODELS = (
    ModelSpec(
        adapter_name='lagged_ridge_forecaster',
        display_name='lagged_ridge_forecaster',
        params={
            'window_size': 18,
            'stride': 1,
            'alpha': 1.0,
            'stage_tuning_runtime': {
                'metric_name': 'rmse',
                'max_values_per_parameter': 2,
                'max_stage_candidates': 6,
            },
        },
    ),
    ModelSpec(
        adapter_name='patch_tst_model',
        display_name='patch_tst_model',
        params={
            'patch_len': 16,
            'epochs': 20,
            'batch_size': 16,
            'learning_rate': 1e-3,
            'activation': 'ReLU',
            'stage_tuning_runtime': {
                'metric_name': 'rmse',
                'max_values_per_parameter': 2,
                'max_stage_candidates': 6,
            },
        },
    ),
)

config = BenchmarkSuiteConfig(
    task_type=TaskType.FORECASTING,
    datasets=M4_DATASETS,
    models=FORECASTING_MODELS,
    metrics=('mase', 'smape', 'owa', 'rmse', 'mae'),
    artifact_spec=ArtifactSpec(
        output_dir=f'benchmark/results/v2_demo/neural_stage_suite_{EXPERIMENT_DATE}',
        persist_on_run=True,
    ),
    run_spec=RunSpec(
        run_name='neural_stage_suite',
        primary_metric='rmse',
        show_progress=True,
        progress_leave=False,
        progress_log_errors=True,
        progress_log_summaries=True,
    ),
)

result = run_forecasting_benchmark_suite(config)
