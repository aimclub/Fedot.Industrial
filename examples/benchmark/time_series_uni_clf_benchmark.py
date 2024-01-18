from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
from benchmark.benchmark_TSC import BenchmarkTSC
from fedot_ind.core.repository.model_repository import default_industrial_availiable_operation

ml_task = 'classification'
available_opearations = default_industrial_availiable_operation(ml_task)
experiment_setup = {'problem': ml_task,
                    'metric': 'accuracy',
                    'timeout': 60,
                    'num_of_generations': 15,
                    'pop_size': 10,
                    'logging_level': 10,
                    'available_operations': available_opearations,
                    'n_jobs': 6,
                    'backend': 'CUDA',
                    'initial_assumption': None,
                    'max_pipeline_fit_time': 10,
                    'with_tuning': False,
                    'tuning_params': {'tuning_timeout': 10,
                                      'tuning_iterations': 1000,
                                      'tuning_early_stop': 50},
                    'industrial_preprocessing': False,
                    'early_stopping_iterations': 5,
                    'early_stopping_timeout': 75,
                    'optimizer': IndustrialEvoOptimizer}

if __name__ == "__main__":
    benchmark = BenchmarkTSC(experiment_setup=experiment_setup,
                             custom_datasets=[
                                 #'FordA',
                                 'HandOutlines',
                                 'NonInvasiveFetalECGThorax2',
                                 'NonInvasiveFetalECGThorax1',
                                 'HouseTwenty',
                                 'OliveOil',
                                 'Beef',
                                 'Phoneme',
                                 'Plane',
                                 'FacesUCR',
                                 'FreezerSmallTrain',
                             ],
                             use_small_datasets=True)
    benchmark.run()
