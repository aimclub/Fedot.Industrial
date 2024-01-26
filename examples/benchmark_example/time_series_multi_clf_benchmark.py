from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
from benchmark.benchmark_TSC import BenchmarkTSC
from fedot_ind.core.repository.model_repository import default_industrial_availiable_operation

ml_task = 'classification'
available_operations = default_industrial_availiable_operation(ml_task)
experiment_setup = {'problem': ml_task,
                    'metric': 'accuracy',
                    'timeout': 180,
                    'num_of_generations': 15,
                    'pop_size': 10,
                    'logging_level': 10,
                    'available_operations': available_operations,
                    'n_jobs': 4,
                    'industrial_preprocessing': False,
                    'initial_assumption': None,
                    'max_pipeline_fit_time': 25,
                    'with_tuning': False,
                    'early_stopping_iterations': 5,
                    'early_stopping_timeout': 60,
                    'optimizer': IndustrialEvoOptimizer}

if __name__ == "__main__":
    benchmark = BenchmarkTSC(experiment_setup=experiment_setup,
                             custom_datasets=[
                                 'SelfRegulationSCP1',
                                 'Handwriting',
                                 'PhonemeSpectra',
                                 'DuckDuckGeese',
                                 'ERing',
                                 'PEMS - SF'
                             ],
                             use_small_datasets=False)
    benchmark.run()
