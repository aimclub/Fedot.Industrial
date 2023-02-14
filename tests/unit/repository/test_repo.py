import numpy as np
from fedot.core.composer.metrics import ROCAUC
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum, Task

from core.repository.initializer_industrial_models import initialize_industrial_models
from tests.unit.api.test_API_config import load_data


def test_repo():
    initialize_industrial_models()
    train_data, test_data, n_classes = load_data('Lightning7')
    train_data = InputData(idx=np.arange(len(train_data[0])),
                           features=train_data[0],
                           target=train_data[1],
                           task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table)
    test_data = InputData(idx=np.arange(len(test_data[0])),
                          features=test_data[0],
                          target=test_data[1],
                          task=Task(TaskTypesEnum.classification),
                          data_type=DataTypesEnum.table)
    pipeline = PipelineBuilder().add_node('data_driven_basic').add_node('topological_extractor').add_node(
        'rf').to_pipeline()
    pipeline.fit(train_data)
    predict = pipeline.predict(test_data)
    print(ROCAUC.metric(test_data, predict))
