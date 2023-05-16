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

from fedot_ind.core.operation.transformation.WindowSelection import WindowSizeSelection
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


train_data, test_data = get_ts_data('australia', 20)

with IndustrialModels():
    n, _ = WindowSizeSelection(time_series=train_data.features,
                               wss_algorithm='dominant_fourier_frequency').get_window_size()
    print(n)
    pipeline = PipelineBuilder().add_node('data_driven_basis_for_forecasting',
                                          params={'n_components': 6, 'window_size': n,
                                                  'estimator': PipelineBuilder().add_node('ar').build()
                                                  },
                                          ).build()
    pipeline2 = PipelineBuilder().add_node('ar').build()
    pipeline_tuner = TunerBuilder(train_data.task) \
        .with_tuner(SimultaneousTuner) \
        .with_metric(RegressionMetricsEnum.MAE) \
        .with_cv_folds(3) \
        .with_validation_blocks(2) \
        .with_iterations(10) \
        .build(train_data)
    pipeline2 = pipeline_tuner.tune(pipeline2)
    pipeline.fit(train_data)
    pipeline2.fit(train_data)
    predict = np.ravel(pipeline.predict(test_data).predict)
    no_ssa = np.ravel(pipeline2.predict(test_data).predict)
    plt.plot(train_data.idx, test_data.features, label='features')
    plt.plot(test_data.idx, test_data.target, label='target')
    plt.plot(test_data.idx, predict, label='predicted ssa')
    plt.plot(test_data.idx, no_ssa, label='predicted no ssa')
    print(f"SSA smape: {smape(test_data.target, predict)}")
    print(f"no SSA smape: {smape(test_data.target, no_ssa)}")
    plt.grid()
    plt.legend()
    plt.show()
