from benchmark.benchmark_TSER import BenchmarkTSER
from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer

experiment_setup = {'problem': 'regression',
                    'metric': 'rmse',
                    'timeout': 40,
                    'num_of_generations': 5,
                    'pop_size': 10,
                    'logging_level': 10,
                    'available_operations':
                        ['rfr',
                         'ridge',
                         'gbr',
                         'sgdr',
                         'linear',
                         'xgbreg',
                         'dtreg',
                         'treg',
                         'knnreg',
                         'scaling',
                         'normalization',
                         'pca'
                         'isolation_forest_reg',
                         'eigen_basis',
                         'fourier_basis',
                         'minirocket_extractor',
                         'quantile_extractor',
                         'signal_extractor',
                         'topological_features'
                         ],
                    'n_jobs': 1,
                    'initial_assumption': None,
                    'max_pipeline_fit_time': 10,
                    'with_tuning': False,
                    'early_stopping_iterations': 5,
                    'early_stopping_timeout': 40,
                    'optimizer': IndustrialEvoOptimizer}

if __name__ == "__main__":
    benchmark = BenchmarkTSER(experiment_setup=experiment_setup,
                              custom_datasets=[
                                  # "AustraliaRainfall",
                                  #"BeijingPM10Quality",
                                  "BeijingPM25Quality",
                                  "BenzeneConcentration",
                                  # "BIDMC32HR",
                                  # "BIDMC32RR",
                                  # "BIDMC32SpO2",
                                  # "Covid3Month",
                                  #"AppliancesEnergy",
                                  # "HouseholdPowerConsumption1",
                                  # "HouseholdPowerConsumption2",
                                  # "IEEEPPG",
                                  "LiveFuelMoistureContent",
                                  "NewsHeadlineSentiment",
                                 # "NewsTitleSentiment",
                                  "PPGDalia",
                              ])
    benchmark.run()

