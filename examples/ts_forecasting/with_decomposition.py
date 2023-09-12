import random

import numpy as np
import pandas as pd
from fedot.core.composer.metrics import smape
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from matplotlib import pyplot as plt

from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

datasets = {
    'm4_yearly': f'../data/ts/M4YearlyTest.csv',
    'm4_weekly': f'../data/ts/M4WeeklyTest.csv',
    'm4_daily': f'../data/ts/M4DailyTest.csv',
    'm4_monthly': f'../data/ts/M4MonthlyTest.csv',
    'm4_quarterly': f'../data/ts/M4QuarterlyTest.csv'}


def get_ts_data(dataset='m4_monthly', horizon: int = 30, m4_id=None):
    time_series = pd.read_csv(datasets[dataset])

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=horizon))
    if not m4_id:
        label = random.choice(np.unique(time_series['label']))
    else:
        label = m4_id
    print(label)
    time_series = time_series[time_series['label'] == label]

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
    train_data, test_data = train_test_data_setup(train_input)
    return train_data, test_data, label


if __name__ == '__main__':

    forecast_length = 13

    train_data, test_data, label = get_ts_data('m4_monthly', forecast_length)

    with IndustrialModels():
        pipeline = PipelineBuilder().add_node('data_driven_basis_for_forecasting',
                                              params={'window_size': int(len(train_data.features) * 0.35)}
                                              ).build()
        pipeline.fit(train_data)
        ssa_predict = np.ravel(pipeline.predict(test_data).predict)

    baseline = PipelineBuilder().add_node('ar').build()
    baseline.fit(train_data)
    no_ssa = np.ravel(baseline.predict(test_data).predict)

    plt.title(label)
    plt.plot(train_data.idx, test_data.features, label='features')
    plt.plot(test_data.idx, test_data.target, label='target')
    plt.plot(test_data.idx, ssa_predict, label='predicted ssa')
    plt.plot(test_data.idx, no_ssa, label='predicted baseline')
    plt.grid()
    plt.legend()
    plt.show()

    print(f"SSA smape: {smape(test_data.target, ssa_predict)}")
    print(f"no SSA smape: {smape(test_data.target, no_ssa)}")
