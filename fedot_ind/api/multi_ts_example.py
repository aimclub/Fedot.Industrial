import os
import sys

import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from matplotlib import pyplot as plt

from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader
from fedot_ind.core.models.signal.RecurrenceExtractor import RecurrenceExtractor
from fedot_ind.core.models.statistical.StatsExtractor import StatsExtractor
from fedot_ind.core.operation.transformation.basis.data_driven import DataDrivenBasisImplementation
from fedot_ind.core.operation.transformation.basis.fourier import FourierBasisImplementation
from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, \
    mean_squared_error, d2_absolute_error_score, \
    median_absolute_error, r2_score


def init_input(X, y):
    input_data = InputData(idx=np.arange(len(X)),
                           features=np.array(X.values.tolist()),
                           target=y.reshape(-1, 1),
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.image)
    input_data.features = np.where(np.isnan(input_data.features), 0, input_data.features)
    input_data.features = np.where(np.isinf(input_data.features), 0, input_data.features)
    return input_data


def prepare_features(dataset_name, pca_n_components: float = 0.9):
    train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()

    train_target = np.array([float(i) for i in train_data[1]])
    test_target = np.array([float(i) for i in test_data[1]])
    train_input_data = init_input(train_data[0], train_target)
    test_input_data = init_input(test_data[0], test_target)

    # extractor = StatsExtractor({'window_mode': False,
    #                             'window_size': 10,
    #                             'var_threshold': 0})
    extractor = RecurrenceExtractor({'window_mode': False,
                                     'min_signal_ratio': 0.3,
                                     'max_signal_ratio': 0.8,
                                     'rec_metric': 'euclidean'
                                     })

    pca = PCA(n_components=pca_n_components,
              svd_solver='full')

    extracted_features_train = extractor.transform(train_input_data)
    train_size = extracted_features_train.features.shape
    train_features = extracted_features_train.features.reshape(train_size[0], train_size[1] * train_size[2])
    train_features = pca.fit_transform(train_features)

    extracted_features_test = extractor.transform(test_input_data)
    test_size = extracted_features_test.features.shape
    test_features = extracted_features_test.features.reshape(test_size[0], test_size[1] * test_size[2])
    test_features = pca.transform(test_features)
    return train_features, train_target, test_features, test_target


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


def evaluate_baseline(train_features, train_target, test_features, test_target):
    node_scaling = PipelineNode('scaling')
    node_rfr = PipelineNode('lasso', nodes_from=[node_scaling])
    baseline_model = Pipeline(node_rfr)
    input_fit = InputData(idx=np.arange(len(train_features)),
                          features=train_features,
                          target=train_target.reshape(-1, 1),
                          task=Task(TaskTypesEnum.regression),
                          data_type=DataTypesEnum.image)
    input_predict = InputData(idx=np.arange(len(test_features)),
                              features=test_features,
                              target=test_target.reshape(-1, 1),
                              task=Task(TaskTypesEnum.regression),
                              data_type=DataTypesEnum.image)

    baseline_model.fit(input_fit)
    labels_baseline = baseline_model.predict(input_predict).predict
    metric_df_baseline = calculate_metric(test_target, labels_baseline)
    return metric_df_baseline

    # ddb_features_train = DataDrivenBasisImplementation({'window_size': 30,
    #                                                     'sv_selector': 'median'}).transform(train_input_data)
    # fourier_features_train = FourierBasisImplementation({"spectrum_type": "smoothed",
    #                                                      "threshold": 20000}).transform(train_input_data)
    # fourier_features_test = FourierBasisImplementation({"spectrum_type": "smoothed",
    #                                                     "threshold": 20000}).transform(test_input_data)


if __name__ == "__main__":
    dataset_list = [
        'AppliancesEnergy',
        # 'AustraliaRainfall',
        # 'BeijingPM10Quality',
        # 'BeijingPM25Quality',
        # 'BenzeneConcentration',
        # 'HouseholdPowerConsumption1',
        'HouseholdPowerConsumption2',
        # 'IEEEPPG',
        # 'FloodModeling1',
        # 'FloodModeling2',
        # 'FloodModeling3'
        # 'LiveFuelMoistureContent',
        'BIDMC32HR',
        'BIDMC32RR',
        'BIDMC32SpO2'
    ]
    ten_minutes = range(0, 3, 1)
    one_hour = ['1hr']
    for dataset_name in dataset_list:
        try:
            os.makedirs(f'./{dataset_name}')
        except Exception:
            _ = 1

        train_features, train_target, test_features, test_target = prepare_features(dataset_name=dataset_name)
        metric_df_baseline = evaluate_baseline(train_features, train_target, test_features, test_target)
        metric_df_baseline.to_csv(f'./{dataset_name}/baseline_metrics.csv')

        for run in ten_minutes:
            predictor = Fedot(problem='regression',
                              timeout=10,
                              metric='rmse',
                              n_jobs=6)
            model = predictor.fit(features=train_features, target=train_target)
            labels = predictor.predict(features=test_features)
            metric_df = calculate_metric(test_target, labels)
            metric_df.to_csv(f'./{dataset_name}/metrics_run_{run}.csv')
            pipeline = predictor.current_pipeline
            pipeline.show(f'./{dataset_name}/pipeline_structure_{run}.png')
            predictor.history.save(f'./{dataset_name}/history_run_{run}.json')
            path_to_save = f'./{dataset_name}/saved_pipelines_run_{run}'
            pipeline.save(path=path_to_save, create_subdir=True, is_datetime_in_path=True)

    _ = 1
