import os

import numpy as np
import pandas as pd
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.tools.example_utils import industrial_common_modelling_loop

dataset_name = 'liman'
power_load = '20%'
failure_type = 'bearing_failure'
train_type = 'without_failure'
initial_assumptions = {'fourier_stat_detector': PipelineBuilder().add_node(
    'fourier_basis').add_node('stat_detector')}


def read_data(path_train, path_test):
    path_train = PROJECT_PATH + '/examples/data/' + path_train
    path_test = PROJECT_PATH + '/examples/data/' + path_test
    train_data = [
        pd.read_csv(
            f'{path_train}/{i}',
            index_col=0).values for i in os.listdir(path_train)]
    train_data = np.concatenate(train_data, axis=1).T
    train_target = np.zeros((train_data.shape[0], 1))
    test_data = [
        pd.read_csv(
            f'{path_test}/{i}',
            index_col=0).values for i in os.listdir(path_test)]
    test_data = np.concatenate(test_data, axis=1).T
    return (train_data, train_target), (test_data, None)


if __name__ == "__main__":
    path_train, path_test = f'{dataset_name}/{train_type}/{power_load}', \
                            f'{dataset_name}/{failure_type}/{power_load}'
    train_data, test_data = read_data(path_train, path_test)
    data_dict = dict(train_data=train_data, test_data=test_data)
    prediction_window = 10
    finetune = False
    # metric_names = ('nab')
    metric_names = ('accuracy')
    api_config = dict(
        problem='classification',
        metric='accuracy',
        timeout=1,
        pop_size=10,
        initial_assumption=initial_assumptions['fourier_stat_detector'],
        industrial_strategy_params={
            'industrial_task': 'anomaly_detection',
            'detection_window': prediction_window,
            'data_type': 'time_series'},
        with_tunig=False,
        n_jobs=2,
        logging_level=20)

    model, labels, metrics = industrial_common_modelling_loop(
        api_config=api_config, dataset_name=data_dict, finetune=finetune, metric_names=metric_names)
    print(metrics)
