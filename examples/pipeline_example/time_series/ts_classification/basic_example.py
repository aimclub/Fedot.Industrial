import matplotlib
from fedot import Fedot
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from examples.example_utils import init_input_data
from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
from fedot_ind.tools.loader import DataLoader
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from statsmodels.tools.eval_measures import rmse

matplotlib.use('TkAgg')
model_dict = {'basic_quantile': PipelineBuilder().add_node('quantile_extractor',
                                                           params={'window_size': 10,
                                                                   'stride': 5}).add_node('rf'),
              'basic_topological': PipelineBuilder().add_node('topological_extractor',
                                                              params={'window_size': 10,
                                                                      'stride': 5}).add_node('rf'),
              'basic_recurrence': PipelineBuilder().add_node('recurrence_extractor',
                                                             params={'window_size': 10,
                                                                     'stride': 5}).add_node('rf'),
              'advanced_quantile': PipelineBuilder().add_node('fourier_basis').add_node('quantile_extractor',
                                                                                        params={
                                                                                            'window_size': 10}).add_node(
                  'rf'),
              'advanced_topological': PipelineBuilder().add_node('eigen_basis').add_node('topological_extractor',
                                                                                         params={'stride': 20,
                                                                                                 'window_size': 10}).add_node(
                  'rf'),
              'advanced_reccurence': PipelineBuilder().add_node('wavelet_basis').add_node(
                  'recurrence_extractor',
                  params={'window_size': 0}).add_node(
                  'rf')
              }

ts_reg_operations = ['rfr',
                     'ridge',
                     'scaling',
                     'normalization',
                     'pca',
                     'xgbreg',
                     'svr',
                     'dtreg',
                     'treg',
                     'knnreg',
                     'kernel_pca',
                     'isolation_forest_reg',
                     'eigen_basis',
                     'fourier_basis',
                     'dimension_reduction',
                     'minirocket_extractor',
                     'quantile_extractor',
                     'signal_extractor',
                     'topological_features'
                     ]

# 14 Regression equal length no missing problems [1]
monash_regression = [
    "AppliancesEnergy",
    "AustraliaRainfall",
    "BIDMC32HR",
    "BIDMC32RR",
    "BIDMC32SpO2",
    "Covid3Month",
    "FloodModeling1",
    "FloodModeling2",
    "FloodModeling3",
    "IEEEPPG",
    "LiveFuelMoistureContent",
    "NewsHeadlineSentiment",
    "NewsTitleSentiment",
    "PPGDalia",
]


if __name__ == "__main__":
    OperationTypesRepository = IndustrialModels().setup_repository()
    problem = 'regression'
    metric = 'rmse'


    error_model = PipelineBuilder().add_node('knnreg').build()
    error_model = PipelineBuilder().add_node('knnreg').build()
    for dataset in monash_regression:
        train_data, test_data = DataLoader(dataset_name=dataset).load_data()
        input_data = init_input_data(train_data[0], train_data[1], task=problem)
        val_data = init_input_data(test_data[0], test_data[1], task=problem)
        model = Fedot(problem=problem,
                      logging_level=10,
                      n_jobs=2,
                      metric=metric,
                      pop_size=20,
                      num_of_generations=20,
                      optimizer=IndustrialEvoOptimizer,
                      available_operations=ts_reg_operations,
                      max_pipeline_fit_time=4,
                      timeout=30,
                      with_tuning=False
                      )
        model = error_model
        model.fit(input_data)
        features = model.predict(val_data)
        f = rmse(val_data.target, features.predict)[0]

    _ = 1
