from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
from benchmark.benchmark_TSC import BenchmarkTSC

experiment_setup = {'problem': 'classification',
                    'metric': 'accuracy',
                    'timeout': 120,
                    'num_of_generations': 15,
                    'pop_size': 20,
                    'available_operations': [
                        'eigen_basis',
                        'dimension_reduction',
                        'inception_model',
                        'logit',
                        'rf',
                        'xgboost',
                        'minirocket_extractor',
                        'normalization',
                        'omniscale_model',
                        'pca',
                        'mlp',
                        'quantile_extractor',
                        'scaling',
                        'signal_extractor',
                        'topological_features'
                    ],
                    'n_jobs': 2,
                    'initial_assumption': None,
                    'max_pipeline_fit_time': 10,
                    'with_tuning': False,
                    'early_stopping_iterations': 5,
                    'early_stopping_timeout': 60,
                    'optimizer': IndustrialEvoOptimizer}

if __name__ == "__main__":
    benchmark = BenchmarkTSC(experiment_setup=experiment_setup,
                             custom_datasets=['Epilepsy'
                                              'EthanolConcentration',
                                              'ERing',
                                              'FaceDetection',
                                              'StandWalkJump',
                                              'UWaveGestureLibrary'
                                              ],
                             use_small_datasets=False)
    benchmark.run()
