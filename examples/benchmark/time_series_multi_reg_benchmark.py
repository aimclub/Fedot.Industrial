from benchmark.benchmark_TSER import BenchmarkTSER
from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
from fedot_ind.core.repository.model_repository import default_industrial_availiable_operation

ml_task = 'regression'
available_opearations = default_industrial_availiable_operation(ml_task)
experiment_setup = {'problem': ml_task,
                    'metric': 'rmse',
                    'timeout': 60,
                    'num_of_generations': 10,
                    'pop_size': 20,
                    'logging_level': 10,
                    'available_operations':
                        available_opearations,
                    'n_jobs': 2,
                    'backend': 'CUDA',
                    'industrial_preprocessing': True,
                    'initial_assumption': None,
                    'max_pipeline_fit_time': 15,
                    'with_tuning': False,
                    'early_stopping_iterations': 5,
                    'early_stopping_timeout': 60,
                    'optimizer': IndustrialEvoOptimizer}

if __name__ == "__main__":
    benchmark = BenchmarkTSER(experiment_setup=experiment_setup,
                              custom_datasets=[
                                  # "AustraliaRainfall",
                                  # "BeijingPM10Quality",
                                  # "BeijingPM25Quality",
                                  # "BenzeneConcentration",
                                  # "BIDMC32HR",
                                  # "BIDMC32RR",
                                  # "BIDMC32SpO2",
                                  # "Covid3Month",
                                  # "AppliancesEnergy",
                                  # "HouseholdPowerConsumption1",
                                  # "HouseholdPowerConsumption2",
                                  # "IEEEPPG",
                                  # "ChilledWaterPredictor",
                                 # "GasSensorArrayEthanol",
                                 # "GasSensorArrayAcetone",
                                 # "CardanoSentiment",
                                  # "BinanceCoinSentiment",
                                  # "SolarRadiationAndalusia",
                                  # "PrecipitationAndalusia",
                                  #"VentilatorPressure",
                                  #"OccupancyDetectionLight",
                                  #"DhakaHourlyAirQuality",
                                  #"WaveDataTension",
                                  #"NaturalGasPricesSentiment",
                                  #"DailyTemperatureLatitude",
                                 #"AluminiumConcentration",
                                 # "CopperConcentration",
                                 # "ZincConcentration",
                                 # "CalciumConcentration",
                                 # "MagnesiumConcentration",
                              ])
    benchmark.run()
