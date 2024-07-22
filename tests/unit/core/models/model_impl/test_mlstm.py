import pytest

from fedot.core.data.data import InputData
from fedot_ind.core.models.nn.network_impl.mlstm import MLSTM
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
import numpy as np


_N_FEATURES = 73
_N_SAMPLES = 133
_N_CLASSES = 3
_INTERVAL_LENGTH = 7

@pytest.fixture
def data():
    X, y = np.random.randn(_N_SAMPLES, _N_FEATURES), np.random.randint(0, _N_CLASSES, size=_N_SAMPLES)
    return InputData(idx=np.arange(0, len(X)),
                            features=X,
                            target=y,
                            task=Task(TaskTypesEnum.classification),
                            data_type=DataTypesEnum.table)

@pytest.mark.parametrize('fitting_mode', ['zero_padding', 'moving_window'])
def test_mlstm_by_mode(data, fitting_mode):
    with IndustrialModels():
        ppl = PipelineBuilder().add_node('mlstm_model', 
                                         params={'epochs': 5, 'fitting_mode': fitting_mode}).build()
        ppl.fit(data)
        pred = ppl.predict(data).predict
        assert not np.isnan(pred).any()




