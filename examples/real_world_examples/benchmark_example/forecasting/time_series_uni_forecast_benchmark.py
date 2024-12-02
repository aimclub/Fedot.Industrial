from benchmark.benchmark_TSF import BenchmarkTSF
from fedot_ind.core.repository.constanst_repository import M4_FORECASTING_BENCH

ml_task = 'ts_forecasting'
experiment_setup = {'problem': ml_task,
                    'metric': 'rmse',
                    'timeout': 60,
                    'num_of_generations': 10,
                    'pop_size': 10,
                    'logging_level': 50,
                    'n_jobs': 4,
                    'max_pipeline_fit_time': 25,
                    'with_tuning': False,
                    'early_stopping_iterations': 5,
                    'early_stopping_timeout': 60}

if __name__ == "__main__":
    benchmark = BenchmarkTSF(experiment_setup=experiment_setup,
                             custom_datasets=M4_FORECASTING_BENCH)
    benchmark.run()
