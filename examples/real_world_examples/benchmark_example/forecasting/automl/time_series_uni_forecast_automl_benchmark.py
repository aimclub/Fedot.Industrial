from benchmark.benchmark_TSF import BenchmarkTSF

ml_task = 'ts_forecasting'
model_list = [
    'ar',
    # 'glm',
    'deepar_model',
    # 'tcn_model',
    'lagged',
    # 'sparse_lagged', 'locf',
    'smoothing',
    'gaussian_filter',
    'ridge',
    'lasso',
    # 'lgbmreg',
    'catboostreg',
    # 'pdl_reg',
    # 'eigen_basis', 'wavelet_basis', 'fourier_basis'
]
experiment_setup = {'problem': ml_task,
                    'metric': 'rmse',
                    'timeout': 2,
                    'num_of_generations': 10,
                    'pop_size': 10,
                    'logging_level': 10,
                    'n_jobs': 1,
                    'optimizer_params': {'mutation_agent': 'bandit',
                                         'mutation_strategy': 'growth_mutation_strategy'},
                    'available_operations': model_list,
                    'max_pipeline_fit_time': 25,
                    'with_tuning': False,
                    'early_stopping_iterations': 5,
                    'early_stopping_timeout': 60}

FORECASTING_BENCH = 'automl_univariate'
path = 'examples/real_world_examples/benchmark_example/forecasting/automl/shell/data/univariate_libra'
if __name__ == "__main__":
    benchmark = BenchmarkTSF(experiment_setup=experiment_setup,
                             custom_datasets=FORECASTING_BENCH)
    benchmark.run(path)
