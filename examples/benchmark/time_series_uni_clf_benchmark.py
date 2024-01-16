from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
from benchmark.benchmark_TSC import BenchmarkTSC

experiment_setup = {'problem': 'classification',
                    'metric': 'accuracy',
                    'timeout': 90,
                    'num_of_generations': 10,
                    'pop_size': 10,
                    'logging_level': 10,
                    'available_operations': [
                        'eigen_basis',
                        'fourier_basis',
                        'wavelet_basis',
                        'inception_model',
                        'logit',
                        'rf',
                        'minirocket_extractor',
                        'normalization',
                        'omniscale_model',
                        'pca',
                        'mlp',
                        'quantile_extractor',
                        'scaling',
                        'signal_extractor'
                    ],
                    'n_jobs': 1,
                    'initial_assumption': None,
                    'max_pipeline_fit_time': 10,
                    'with_tuning': False,
                    'tuning_params': {'tuning_timeout': 10,
                                      'tuning_iterations': 1000,
                                      'tuning_early_stop': 50},
                    'industrial_preprocessing': False,
                    'early_stopping_iterations': 5,
                    'early_stopping_timeout': 60,
                    'optimizer': IndustrialEvoOptimizer}

if __name__ == "__main__":
    benchmark = BenchmarkTSC(experiment_setup=experiment_setup,
                             custom_datasets=[
                                 # 'Lightning7',
                                 # 'SmoothSubspace',
                                 # 'FordA',
                                 # 'UWaveGestureLibraryAll',
                                 # 'BeetleFly',
                                 # # 'DistalPhalanxOutlineCorrect',
                                 # # 'CinCECGTorso',
                                 # # 'ECG5000',
                                 # # 'UWaveGestureLibraryY',
                                 # # 'Wafer',
                                 # # 'ProximalPhalanxOutlineCorrect',
                                 # # 'Earthquakes',
                                 # #  'ShapeletSim',
                                 # 'CBF',
                                 # 'ChlorineConcentration',
                                 'DistalPhalanxOutlineAgeGroup',
                                 'Plane',
                                 'FacesUCR',
                                 'FreezerSmallTrain',
                             ],
                             use_small_datasets=True)
    # benchmark.create_report()
    # benchmark.finetune()
    benchmark.run()
