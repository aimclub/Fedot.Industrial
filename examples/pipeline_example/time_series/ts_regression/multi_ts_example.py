import os

from fedot_ind.core.architecture.settings.computational import backend_methods as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, \
    mean_squared_error, d2_absolute_error_score, \
    median_absolute_error, r2_score

from fedot_ind.tools.loader import DataLoader
from fedot_ind.core.models.quantile.quantile_extractor import QuantileExtractor
from fedot_ind.core.models.recurrence.reccurence_extractor import RecurrenceExtractor

model_dict = {'regression_with_statistical_features': PipelineBuilder().add_node('data_driven_basis_for_forecasting',
                                                                                 params={'window_size': 10,
                                                                                         }
                                                                                 ),
              'regression_with_reccurence_features': PipelineBuilder().add_node('recurrence_extractor')}

datasets = {
    'm4_yearly': f'../data/ts/M4YearlyTest.csv',
    'm4_weekly': f'../data/ts/M4WeeklyTest.csv',
    'm4_daily': f'../data/ts/M4DailyTest.csv',
    'm4_monthly': f'../data/ts/M4MonthlyTest.csv',
    'm4_quarterly': f'../data/ts/M4QuarterlyTest.csv'}

forecast_length = 13


def init_input(X, y):
    input_data = InputData(idx=np.arange(len(X)),
                           features=np.array(X.values.tolist()),
                           target=y.reshape(-1, 1),
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.image)
    input_data.features = np.where(np.isnan(input_data.features), 0, input_data.features)
    input_data.features = np.where(np.isinf(input_data.features), 0, input_data.features)
    return input_data


def prepare_features(dataset_name,
                     pca_n_components: float = 0.95,
                     feature_generator: list = ['statistical'],
                     reduce_dimension: bool = False):
    train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()

    train_target = np.array([float(i) for i in train_data[1]])
    test_target = np.array([float(i) for i in test_data[1]])
    train_input_data = init_input(train_data[0], train_target)
    test_input_data = init_input(test_data[0], test_target)

    generator_dict = {'statistical': QuantileExtractor({'window_mode': False,
                                                        'window_size': 10,
                                                        'var_threshold': 0}),
                      'reccurence': RecurrenceExtractor({'window_mode': True,
                                                         'min_signal_ratio': 0.7,
                                                         'max_signal_ratio': 0.9,
                                                         'rec_metric': 'cosine'
                                                         })
                      }
    train_features_list, test_features_list = [], []
    for extractor in feature_generator:
        extractor = generator_dict[extractor]
        extracted_features_train = extractor.transform(train_input_data)
        train_size = extracted_features_train.predict.shape
        train_features = extracted_features_train.predict

        extracted_features_test = extractor.transform(test_input_data)
        test_size = extracted_features_test.predict.shape
        test_features = extracted_features_test.predict

        if reduce_dimension:
            pca = PCA(n_components=pca_n_components,
                      svd_solver='full')
            train_features = pca.fit_transform(train_features)
            test_features = pca.transform(test_features)
        train_features_list.append(train_features), test_features_list.append(test_features)

    return train_features_list, train_target, test_features_list, test_target


def calculate_metric(test_target, labels):
    metric_dict = {'r2_score:': r2_score(test_target, labels),
                   'mean_squared_error:': mean_squared_error(test_target, labels),
                   'root_mean_squared_error:': np.sqrt(mean_squared_error(test_target, labels)),
                   'mean_absolute_error': mean_absolute_error(test_target, labels),
                   'median_absolute_error': median_absolute_error(test_target, labels),
                   'explained_variance_score': explained_variance_score(test_target, labels),
                   'max_error': max_error(test_target, labels),
                   # 'mean_poisson_deviance': mean_poisson_deviance(test_target, labels),
                   # 'mean_gamma_deviance': mean_gamma_deviance(test_target, labels),
                   'd2_absolute_error_score': d2_absolute_error_score(test_target, labels)
                   }
    df = pd.DataFrame.from_dict(metric_dict, orient='index')
    return df


def evaluate_baseline(train, train_target, test, test_target):
    node_scaling = PipelineNode('scaling')
    node_rfr = PipelineNode('lasso', nodes_from=[node_scaling])
    baseline_model = Pipeline(node_rfr)
    input_fit = InputData(idx=np.arange(len(train)),
                          features=train,
                          target=train_target.reshape(-1, 1),
                          task=Task(TaskTypesEnum.regression),
                          data_type=DataTypesEnum.image)
    input_predict = InputData(idx=np.arange(len(test)),
                              features=test,
                              target=test_target.reshape(-1, 1),
                              task=Task(TaskTypesEnum.regression),
                              data_type=DataTypesEnum.image)

    baseline_model.fit(input_fit)
    labels_baseline = baseline_model.predict(input_predict).predict
    metric_df_baseline = calculate_metric(test_target, labels_baseline)
    return metric_df_baseline


if __name__ == "__main__":
    dataset_list = [
        # 'Gazprom',
        # 'AppliancesEnergy',
        # 'AustraliaRainfall',
        # 'BeijingPM10Quality',
        # 'BeijingPM25Quality',
        # 'BenzeneConcentration',
        # 'HouseholdPowerConsumption1',
        # 'HouseholdPowerConsumption2',
        # 'IEEEPPG',
        # 'FloodModeling1',
        'FloodModeling2',
        'FloodModeling3'
        'LiveFuelMoistureContent',
        'BIDMC32HR',
        'BIDMC32RR',
        'BIDMC32SpO2',
        'DailyOilGasPrices',
        'ElectricityPredictor',
        'OccupancyDetectionLight',
        'SolarRadiationAndalusia',
        'TetuanEnergyConsumption',
        'WindTurbinePower',
        'ElectricMotorTemperature',
        'LPGasMonitoringHomeActivity',
        'GasSensorArrayAcetone',
        'GasSensorArrayEthanol',
        'WaveTensionData'

    ]
    ten_minutes = range(0, 3, 1)
    one_hour = ['1hr']
    for dataset_name in dataset_list:
        try:
            os.makedirs(f'./{dataset_name}')
        except Exception:
            _ = 1

        train_features, train_target, test_features, test_target = prepare_features(dataset_name=dataset_name,
                                                                                    reduce_dimension=False,
                                                                                    feature_generator=[
                                                                                        'statistical'
                                                                                        # ,'reccurence'
                                                                                    ])
        if len(train_features) > 1:
            concatenate_train = np.concatenate(train_features, axis=1)
            concatenate_test = np.concatenate(test_features, axis=1)
            train_features.append(concatenate_train)
            test_features.append(concatenate_test)
        else:
            concatenate_train = train_features[0]
            concatenate_test = test_features[0]

        for train, test in zip(train_features, test_features):
            metric_df_baseline = evaluate_baseline(train, train_target, test, test_target)
            print(metric_df_baseline)
        metric_df_baseline.to_csv(f'./{dataset_name}/baseline_metrics.csv')

        for run in one_hour:
            predictor = Fedot(problem='regression',
                              metric='rmse',
                              timeout=60,
                              early_stopping_timeout=30,
                              logging_level=20,
                              n_jobs=6)
            model = predictor.fit(features=concatenate_train, target=train_target)
            labels = predictor.predict(features=concatenate_test)
            metric_df = calculate_metric(test_target, labels)
            metric_df.to_csv(f'./{dataset_name}/metrics_run_{run}.csv')
            pipeline = predictor.current_pipeline
            pipeline.show(f'./{dataset_name}/pipeline_structure_{run}.png')
            predictor.history.save(f'./{dataset_name}/history_run_{run}.json')
            path_to_save = f'./{dataset_name}/saved_pipelines_run_{run}'
            pipeline.save(path=path_to_save, create_subdir=True, is_datetime_in_path=True)

    _ = 1
