from benchmark.benchmark_TSER import BenchmarkTSER


experiment_setup = {'problem': 'regression',
                    'metric': 'rmse',
                    'timeout': 10,
                    'num_of_generations': 10,
                    'pop_size': 20,
                    'logging_level': 10,
                    'n_jobs': 2,
                    'industrial_preprocessing': True,
                    'max_pipeline_fit_time': 15,
                    'with_tuning': False,
                    'early_stopping_iterations': 5,
                    'early_stopping_timeout': 60}

if __name__ == "__main__":
    benchmark = BenchmarkTSER(experiment_setup=experiment_setup,
                              custom_datasets=[
                                  "AppliancesEnergy"
                              ])
    benchmark.run()
