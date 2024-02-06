from benchmark.benchmark_TSC import BenchmarkTSC

experiment_setup = {'problem': 'classification',
                    'metric': 'accuracy',
                    'timeout': 60,
                    'num_of_generations': 15,
                    'pop_size': 10,
                    'logging_level': 10,
                    'RAF_workers': 4,
                    'n_jobs': 2,
                    'industrial_preprocessing': True,
                    'max_pipeline_fit_time': 25,
                    'with_tuning': False,
                    'early_stopping_iterations': 5,
                    'early_stopping_timeout': 100}

if __name__ == "__main__":
    benchmark = BenchmarkTSC(experiment_setup=experiment_setup,
                             custom_datasets=[
                                 # 'SelfRegulationSCP1',
                                 # 'Handwriting',
                                 # 'MotorImagery',
                                 # 'EthanolConcentration',
                                 # 'PhonemeSpectra',
                                 # 'DuckDuckGeese',
                                 'ERing',
                                 # 'PEMS - SF'
                             ],
                             use_small_datasets=False)
    benchmark.run()
    _ = 1
