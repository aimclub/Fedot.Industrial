import numpy as np
import pytest
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels


@pytest.fixture(scope='session')
def ts_input_data():
    task = Task(TaskTypesEnum.classification)
    features = np.random.randn(100, 100)
    target = np.random.randint(0, 2, 100)
    train_input = InputData(idx=np.arange(0, 100),
                            features=features,
                            target=target,
                            task=task,
                            data_type=DataTypesEnum.table)
    return train_test_data_setup(train_input, validation_blocks=None)


def test_omniscale_model(ts_input_data):
    train, test = ts_input_data

    with IndustrialModels():
        model = PipelineBuilder().add_node('omniscale_model',
                                           params=dict(epochs=10)
                                           ).build()

        model.fit(train)
        model.predict(test)
        assert model is not None
