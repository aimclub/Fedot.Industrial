import random

import numpy as np
import pandas as pd
from fedot.core.composer.metrics import smape
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from golem.core.tuning.simultaneous import SimultaneousTuner
from matplotlib import pyplot as plt
from statsforecast.models import AutoTheta, AutoARIMA, AutoETS

from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

datasets = {
    'm4_yearly': f'../data/ts/M4YearlyTest.csv',
    'm4_weekly': f'../data/ts/M4WeeklyTest.csv',
    'm4_daily': f'../data/ts/M4DailyTest.csv',
    'm4_monthly': f'../data/ts/M4MonthlyTest.csv',
    'm4_quarterly': f'../data/ts/M4QuarterlyTest.csv'}


def get_ts_data(dataset='australia', horizon: int = 30, m4_id=None):
    time_series = pd.read_csv(datasets[dataset])

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=horizon))
    if 'm4' in dataset:
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
    return train_data, test_data


# , 'M39015'
train_data, test_data = get_ts_data('m4_monthly', 7)

with IndustrialModels():
    pipeline = PipelineBuilder().add_node('data_driven_basis_for_forecasting',
                                          params={'window_size': int(len(train_data.features) * 0.35)}
                                          ).build()

    pipeline.fit(train_data)
    predict = np.ravel(pipeline.predict(test_data).predict)
    model_theta = AutoTheta()
    model_arima = AutoARIMA()
    model_ets = AutoETS()
    forecast = []
    for model in [model_theta, model_arima, model_ets]:
        model.fit(train_data.features)
        p = model.predict(test_data.task.task_params.forecast_length)['mean']
        forecast.append(p)
    pred = np.median(np.array(forecast), axis=0)

    no_ssa = np.ravel(pred)
    plt.plot(train_data.idx, test_data.features, label='features')
    plt.plot(test_data.idx, test_data.target, label='target')
    # for comp in pipeline.nodes[0].fitted_operation.train_basis:
    plt.plot(train_data.idx, np.array(pipeline.nodes[0].fitted_operation.train_basis).sum(axis=0), label='reconstructed features')
    plt.plot(test_data.idx, predict, label='predicted ssa')
    plt.plot(test_data.idx, no_ssa, label='predicted no ssa')
    print(f"SSA smape: {smape(test_data.target, predict)}")
    print(f"no SSA smape: {smape(test_data.target, no_ssa)}")
    plt.grid()
    plt.legend()
    plt.show()
    pipeline.print_structure()
