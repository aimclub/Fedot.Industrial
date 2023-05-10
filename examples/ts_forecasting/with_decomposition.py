import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from matplotlib import pyplot as plt

from fedot_ind.core.architecture.utils.utils import PROJECT_PATH
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

datasets = {
    'australia': f'../data/ts/australia.csv',
    'beer': f'../data/ts/beer.csv',
    'salaries': f'../data/ts/salaries.csv',
    'stackoverflow': f'../data/ts/stackoverflow.csv'}


def get_ts_data(dataset='australia', horizon: int = 30, validation_blocks=None):
    time_series = pd.read_csv(datasets[dataset])

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=horizon))
    if dataset not in ['australia']:
        idx = pd.to_datetime(time_series['idx'].values)
    else:
        # non datetime indexes
        idx = time_series['idx'].values
    time_series = time_series['value'].values
    train_input = InputData(idx=idx,
                            features=time_series,
                            target=time_series,
                            task=task,
                            data_type=DataTypesEnum.ts)
    train_data, test_data = train_test_data_setup(train_input, validation_blocks=validation_blocks)
    return train_data, test_data


train_data, test_data = get_ts_data('australia', 10)

with IndustrialModels():
    pipeline = PipelineBuilder().add_node('data_driven_basis_for_forecasting', params={'n_components': 50}).build()
    pipeline.fit(train_data)

    predict = np.ravel(pipeline.predict(test_data).predict)

    plt.plot(train_data.idx, test_data.features, label='features')
    plt.plot(test_data.idx, test_data.target, label='target')
    plt.plot(test_data.idx, predict, label='predicted')
    plt.grid()
    plt.legend()
    plt.show()
