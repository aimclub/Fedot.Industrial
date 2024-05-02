from benchmark.benchmark_TSER import BenchmarkTSER

experiment_setup = {'problem': 'regression',
                    'metric': 'rmse',
                    'timeout': 100,
                    'num_of_generations': 10,
                    'pop_size': 20,
                    'logging_level': 10,
                    'n_jobs': 2,
                    'early_stopping_iterations': 10,
                    'early_stopping_timeout': 75}

if __name__ == "__main__":
    benchmark = BenchmarkTSER(experiment_setup=experiment_setup,
                              custom_datasets=[
                                  # 'ElectricMotorTemperature',
                                  #              'PrecipitationAndalusia',
                                  #  'AcousticContaminationMadrid',
                                  # 'WindTurbinePower',
                                  # 'DailyOilGasPrices',
                                  # 'DailyTemperatureLatitude',
                                  # 'LPGasMonitoringHomeActivity',
                                  # 'AluminiumConcentration',
                                  # 'BoronConcentration',
                                  # 'CopperConcentration',
                                  # # 'IronConcentration',
                                  #  'ManganeseConcentration',
                                  #  'SodiumConcentration',
                                  #  'PhosphorusConcentration',
                                  #  'PotassiumConcentration',
                                  'MagnesiumConcentration',
                                  'SulphurConcentration',
                                  'ZincConcentration',
                                  'CalciumConcentration'
                              ])
    benchmark.finetune()
    _ = 1
