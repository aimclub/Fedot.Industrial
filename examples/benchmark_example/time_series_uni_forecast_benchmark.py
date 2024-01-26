from fedot.core.repository.tasks import TsForecastingParams

from benchmark.benchmark_TSF import BenchmarkTSF
from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
from benchmark.benchmark_TSC import BenchmarkTSC
from fedot_ind.core.repository.model_repository import default_industrial_availiable_operation

ml_task = 'ts_forecasting'
available_operations = default_industrial_availiable_operation(ml_task)
experiment_setup = {'problem': ml_task,
                    'metric': 'rmse',
                    'task_params': TsForecastingParams(forecast_length=14),
                    'timeout': 60,
                    'num_of_generations': 10,
                    'pop_size': 10,
                    'logging_level': 0,
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
    benchmark = BenchmarkTSF(experiment_setup=experiment_setup,
                             custom_datasets=[
                                 'M3_Monthly_M10'
                             ])
    benchmark.run()
