# import pandas as pd
# from fedot.core.pipelines.pipeline_builder import PipelineBuilder
# from sklearn.metrics import accuracy_score
#
# from examples.benchmark.time_series_multi_reg_benchmark import evaluate_loop
# from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
# from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
# from tsml_eval._wip.results.results_by_classifier import *
#
# available_operations = [
#     'eigen_basis',
#     'dimension_reduction',
#     'inception_model',
#     'logit',
#     'rf',
#     'xgboost',
#     'minirocket_extractor',
#     'normalization',
#     'omniscale_model',
#     'pca',
#     'mlp',
#     'quantile_extractor',
#     'scaling',
#     'signal_extractor',
#     'topological_features'
# ]
# OperationTypesRepository = IndustrialModels().setup_repository()
# initial_assumption = PipelineBuilder().add_node('quantile_extractor').add_node('rf').build()
# experiment_setup = {'task': 'classification',
#                     'metric': 'accuracy',
#                     'timeout': 60,
#                     'available_operations': available_operations,
#                     'n_jobs': 2,
#                     'initial_assumption': initial_assumption,
#                     'max_pipeline_fit_time': 4,
#                     'optimizer': IndustrialEvoOptimizer}
#
# dataset_list = multivariate_equal_length
# metric_dict = {}
# experiment_setup = {'problem': 'regression',
#                     'metric': 'rmse',
#                     'timeout': 1,
#                     'num_of_generations': 10,
#                     'pop_size': 10,
#                     'available_operations':
#                         ['rfr',
#                          'ridge',
#                          'gbr',
#                          'sgdr',
#                          'linear',
#                          'xgbreg',
#                          'dtreg',
#                          'treg',
#                          'knnreg',
#                          'scaling',
#                          'normalization',
#                          'pca'
#                          'isolation_forest_reg',
#                          'eigen_basis',
#                          'fourier_basis',
#                          'minirocket_extractor',
#                          'quantile_extractor',
#                          'signal_extractor',
#                          'topological_features'
#                          ],
#                     'n_jobs': 4,
#                     'initial_assumption': None,
#                     'max_pipeline_fit_time': 10,
#                     'optimizer': IndustrialEvoOptimizer}
#
if __name__ == "__main__":
    benchmark = BenchmarkTSC(experiment_setup=experiment_setup,
                              custom_datasets=["AppliancesEnergy",
                                               "HouseholdPowerConsumption1",
                                               "HouseholdPowerConsumption2",
                                               "IEEEPPG",
                                               "LiveFuelMoistureContent",
                                               "NewsHeadlineSentiment",
                                               "NewsTitleSentiment",
                                               "PPGDalia",
                                               ])
    benchmark.run()
