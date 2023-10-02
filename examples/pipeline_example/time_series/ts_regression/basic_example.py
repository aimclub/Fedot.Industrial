import os
import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, \
    mean_squared_error, d2_absolute_error_score, \
    median_absolute_error, r2_score
from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from examples.example_utils import init_input_data

model_dict = {
    'regression_with_statistical_features': PipelineBuilder().add_node('quantile_extractor',
                                                                       params={'window_size': 5}).add_node('ridge'),
    'regression_pca_with_statistical_features': PipelineBuilder().add_node('quantile_extractor',
                                                                           params={'window_size': 5}).
    add_node('pca', params={'n_components': 0.9}).add_node('ridge'),
    'regression_with_reccurence_features': PipelineBuilder().add_node('recurrence_extractor').add_node('ridge'),
    'regression_pca_with_reccurence_features': PipelineBuilder().add_node('recurrence_extractor').
    add_node('pca', params={'n_components': 0.9}).add_node('ridge')
}
metric_dict = {}

data_path = PROJECT_PATH + '/examples/data'
dataset_list = [
    # 'AppliancesEnergy',
    # 'AustraliaRainfall',
    # 'BeijingPM10Quality',
    # 'BeijingPM25Quality',
    # 'BenzeneConcentration',
    # 'HouseholdPowerConsumption1',
    # 'HouseholdPowerConsumption2',
    # 'IEEEPPG',
    # 'FloodModeling1',
    # 'FloodModeling2',
    # 'FloodModeling3'
    # 'LiveFuelMoistureContent',
    # 'BIDMC32HR',
    # 'BIDMC32RR',
    # 'BIDMC32SpO2',
    # 'DailyOilGasPrices',
    # 'ElectricityPredictor',
    # 'OccupancyDetectionLight',
    # 'SolarRadiationAndalusia',
    # 'TetuanEnergyConsumption',
    # 'WindTurbinePower',
    # 'ElectricMotorTemperature',
    # 'LPGasMonitoringHomeActivity',
    # 'GasSensorArrayAcetone',
    # 'GasSensorArrayEthanol',
    # 'WaveTensionData'
]


def calculate_metric(test_target, labels):
    test_target = test_target.astype(np.float)
    metric_dict = {'r2_score:': r2_score(test_target, labels),
                   'mean_squared_error:': mean_squared_error(test_target, labels),
                   'root_mean_squared_error:': np.sqrt(mean_squared_error(test_target, labels)),
                   'mean_absolute_error': mean_absolute_error(test_target, labels),
                   'median_absolute_error': median_absolute_error(test_target, labels),
                   'explained_variance_score': explained_variance_score(test_target, labels),
                   'max_error': max_error(test_target, labels),
                   'd2_absolute_error_score': d2_absolute_error_score(test_target, labels)
                   }
    df = pd.DataFrame.from_dict(metric_dict, orient='index')
    return df


if __name__ == "__main__":
    dataset_name = 'ElectricityPredictor'
    with IndustrialModels():
        _, train_data, test_data = DataLoader(dataset_name=dataset_name).read_train_test_files(
            dataset_name=dataset_name,
            data_path=data_path)
        for model in model_dict.keys():
            pipeline = model_dict[model].build()
            input_data = init_input_data(train_data[0], train_data[1], task='regression')
            val_data = init_input_data(test_data[0], test_data[1], task='regression')
            pipeline.fit(input_data)
            features = pipeline.predict(val_data).predict
            metric = calculate_metric(test_target=test_data[1], labels=features)
            metric_dict.update({model: metric})
    print(metric_dict)
