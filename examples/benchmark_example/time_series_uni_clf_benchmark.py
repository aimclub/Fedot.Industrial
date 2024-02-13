from benchmark.benchmark_TSC import BenchmarkTSC

experiment_setup = {'problem': 'classification',
                    'metric': 'accuracy',
                    'timeout': 60,
                    'num_of_generations': 15,
                    'pop_size': 10,
                    'logging_level': 10,
                    'n_jobs': 2,
                    'max_pipeline_fit_time': 10,
                    'early_stopping_iterations': 5,
                    'early_stopping_timeout': 45}

if __name__ == "__main__":
    benchmark = BenchmarkTSC(experiment_setup=experiment_setup,
                             custom_datasets=[
                                 # 'ECG200',
                                 # 'Fungi',
                                 # 'InsectEPGRegularTrain',
                                 # 'InsectWingbeatSound',
                                 # 'StarlightCurves',
                                 #'ToeSegmentation1',
                                 # #'ToeSegmentation2',
                                 # 'UMD',
                                 # 'MiddlePhalanxTW',
                                 # 'WormsTwoClass',
                                 # 'Adiac',
                                 # 'Wine'
                             ],
                             use_small_datasets=True)
    benchmark.run()
    _ = 1
