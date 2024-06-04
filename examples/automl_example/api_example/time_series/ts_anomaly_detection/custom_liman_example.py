import os

import numpy as np
import pandas as pd
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.tools.example_utils import industrial_common_modelling_loop

dataset_name = 'liman'
power_load = '20%'
failure_type = 'stator_failure'
train_type = 'without_failure'
initial_assumptions = {
    'fourier_iforest_detector': PipelineBuilder().add_node(
        'fourier_basis').add_node('iforest_detector'),
    'iforest_detector': PipelineBuilder().add_node('iforest_detector')
}
sampling_rate = 4096


def read_data(path_train, path_test, sampling_rate: int = 4096):
    path_train = PROJECT_PATH + '/examples/data/' + path_train
    path_test = PROJECT_PATH + '/examples/data/' + path_test
    train_data = [
        pd.read_csv(
            f'{path_train}/{i}',
            index_col=0, header=None).values for i in os.listdir(path_train)]
    train_data = np.concatenate(train_data, axis=1)
    n_samples, n_channels = round(train_data.shape[0] / sampling_rate), train_data.shape[1]
    train_data = train_data.reshape((n_samples, n_channels, sampling_rate))
    train_target = np.zeros((train_data.shape[0], 1))
    test_data = [
        pd.read_csv(
            f'{path_test}/{i}',
            index_col=0, header=None).values for i in os.listdir(path_test)]
    test_data = np.concatenate(test_data, axis=1)
    test_data = test_data.reshape((n_samples, n_channels, sampling_rate))
    test_target = np.ones((test_data.shape[0], 1))
    return (train_data, train_target), (test_data, test_target)


if __name__ == "__main__":
    failure_type = ['bearing_failure', 'rotor_failure', 'stator_failure']
    power_load = ['100%', '20%', '40%', '60%', '80%', '100%']
    prediction_window = 10
    finetune = False
    # metric_names = ('nab')
    metric_names = ('accuracy')
    for failure in failure_type:
        for load in power_load:
            path_train, path_test = f'{dataset_name}/{train_type}/{load}', \
                                    f'{dataset_name}/{failure}/{load}'
            train_data, test_data = read_data(path_train, path_test)
            data_dict = dict(train_data=train_data, test_data=test_data)
            api_config = dict(
                problem='classification',
                metric='accuracy',
                timeout=0.1,
                pop_size=10,
                cv_folds=2,
                with_tuning=False,
                initial_assumption=initial_assumptions['fourier_iforest_detector'],
                industrial_strategy='anomaly_detection',
                industrial_strategy_params={
                    'sampling_rate': sampling_rate,
                    'detection_window': prediction_window,
                    'data_type': 'tensor'},
                with_tunig=False,
                n_jobs=2,
                logging_level=50)

            model, labels, metrics = industrial_common_modelling_loop(
                api_config=api_config, dataset_name=data_dict, finetune=finetune, metric_names=metric_names)
            print(f'FAILURE_TYPE - {failure}. POWER_LOAD - {load}')
            print(f'___________________________________________________')
            print(f'ACCURACY_OF_DETECTION - {metrics.values[0]}')
            print(f'___________________________________________________')
