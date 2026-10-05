import numpy as np
import pytest
from fedot.core.data.input_data.data import InputData
from fedot.core.data.split.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

from fedot_ind.integration.fedot.extensions import industrial_extension_scope
from fedot_ind.integration.fedot.compatibility import ensure_fedot_tensor_data


@pytest.fixture(scope='session')
def ts():
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


def test_tsc_model(ts):
    train, test = ts
    train = ensure_fedot_tensor_data(train, fit_stage=True)
    test = ensure_fedot_tensor_data(test, fit_stage=False, reference_data=train)
    with industrial_extension_scope():
        pipeline = PipelineBuilder().add_node(
            'tcn_model', params={'epochs': 10}).build()
        pipeline.fit(train)
        predict = pipeline.predict(test)

    assert np.asarray(predict.predict).size == 5
