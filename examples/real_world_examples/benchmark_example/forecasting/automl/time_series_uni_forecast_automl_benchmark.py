from benchmark.benchmark_TSF import BenchmarkTSF

ml_task = 'ts_forecasting'
model_list = [
    'ar',
    'glm',
    'deepar_model',
    'tcn_model',
    # 'lagged',
    # 'sparse_lagged', 'locf',
    # 'smoothing',
    # 'gaussian_filter',
    # 'ridge',
    # 'lasso',
    # 'lgbmreg',
    'catboostreg',
    # 'pdl_reg',
    # 'eigen_basis', 'wavelet_basis', 'fourier_basis'
]

dataset_list = [
    # 'economics',
    'finance',
    # 'human',
    # 'nature'
]

for model in model_list:
    for dataset in dataset_list:
        experiment_setup = {'problem': ml_task,
                            'metric': 'rmse',
                            'timeout': 2,
                            'num_of_generations': 10,
                            'pop_size': 10,
                            'logging_level': 10,
                            'n_jobs': 1,
                            'optimizer_params': {'mutation_agent': 'bandit',
                                                'mutation_strategy': 'growth_mutation_strategy'},
                            'available_operations': [model],
                            'max_pipeline_fit_time': 25,
                            'with_tuning': False,
                            'early_stopping_iterations': 5,
                            'early_stopping_timeout': 60,
                            'output_folder': f'benchmark/results/{model}/{dataset}',
                            'validation_blocks': 1,
                            'cv_folds': None,
                            'was_optimised': True,
                            'is_finetuned': True}

        FORECASTING_BENCH = 'automl_univariate'
        path = f'examples/real_world_examples/benchmark_example/forecasting/automl/data/univariate_libra/{dataset}'
        if __name__ == "__main__":
            benchmark = BenchmarkTSF(experiment_setup=experiment_setup,
                                    custom_datasets=FORECASTING_BENCH)
            benchmark.run(path)
