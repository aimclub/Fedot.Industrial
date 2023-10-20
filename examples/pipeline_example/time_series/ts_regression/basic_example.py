from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot_ind.tools.loader import DataLoader
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from examples.example_utils import init_input_data, calculate_regression_metric

model_dict = {
    'regression_with_statistical_features': PipelineBuilder().add_node('quantile_extractor',
                                                                       params={'window_size': 5}).add_node('ridge'),
    'regression_pca_with_statistical_features': PipelineBuilder().add_node('quantile_extractor',
                                                                           params={'window_size': 5}).
    add_node('pca', params={'n_components': 0.9}).add_node('ridge'),
    'regression_with_reccurence_features': PipelineBuilder().add_node('recurrence_extractor',
                                                                           params={'window_size': 20}).add_node('ridge'),
    'regression_pca_with_reccurence_features': PipelineBuilder().add_node('recurrence_extractor',
                                                                           params={'window_size': 20}).
    add_node('pca', params={'n_components': 0.9}).add_node('ridge'),
    'regression_with_topological_features': PipelineBuilder().add_node('topological_extractor',
                                                                           params={'window_size': 20}).
    add_node('pca', params={'n_components': 0.9}).add_node('ridge'),
    'regression_pca_with_topological_features': PipelineBuilder().add_node('topological_extractor',
                                                                           params={'window_size': 20}).
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

if __name__ == "__main__":
    dataset_name = 'MadridPM10Quality-no-missing'
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
            metric = calculate_regression_metric(test_target=test_data[1], labels=features)
            metric_dict.update({model: metric})
    print(metric_dict)
