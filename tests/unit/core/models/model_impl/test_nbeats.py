import numpy as np
import pytest
from fedot.core.data.input_data.data import InputData
from fedot.core.data.split.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

from fedot_ind.integration.fedot.extensions import industrial_extension_scope


@pytest.fixture(scope='session')
def ts_input_data():
    horizon = 5
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=horizon))
    series = np.random.rand(100)
    train_input = InputData(idx=np.arange(0, len(series)),
                            features=series,
                            target=series,
                            task=task,
                            data_type=DataTypesEnum.ts)
    return train_test_data_setup(train_input, validation_blocks=None)


def nbeats_model(ts_input_data):
    train, test = ts_input_data
    with industrial_extension_scope():
        model = PipelineBuilder().add_node(
            'nbeats_model',
            params={
                'backcast_length': 10,
                'forecast_length': 5,
                'epochs': 10,
            },
        ).build()
        model.fit(train)
        forecast = model.predict(test)

    assert len(forecast.predict) == 5
