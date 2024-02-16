from benchmark.benchmark_TSER import BenchmarkTSER


experiment_setup = {'problem': 'regression',
                    'metric': 'rmse',
                    'timeout': 100,
                    'num_of_generations': 10,
                    'pop_size': 20,
                    'logging_level': 50,
                    'n_jobs': 2,
                    'industrial_preprocessing': False,
                    'max_pipeline_fit_time': 15,
                    'with_tuning': True,
                    'early_stopping_iterations': 10,
                    'early_stopping_timeout': 75}

if __name__ == "__main__":
    benchmark = BenchmarkTSER(experiment_setup=experiment_setup,
                              custom_datasets=[
                                  'DhakaHourlyAirQuality',
                                  'OccupancyDetectionLight',
                                  # 'IEEEPPG',
                                  # 'MethaneMonitoringHomeActivity',
                                  # 'ElectricityPredictor',
                                  # 'EthereumSentiment',
                                  # 'BitcoinSentiment',
                                  # 'SierraNevadaMountainsSnow',
                                  # 'GasSensorArrayAcetone',
                                  'GasSensorArrayEthanol',
                                  'MadridPM10Quality'
                              ])
    benchmark.run()
    _ = 1