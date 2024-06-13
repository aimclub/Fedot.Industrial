import pytest

from fedot_ind.core.architecture.pipelines.abstract_pipeline import AbstractPipeline
from fedot_ind.core.repository.constanst_repository import VALID_LINEAR_CLF_PIPELINE

UNI_MULTI_CLF = ['Earthquakes', 'ERing']
TASK = 'classification'


@pytest.mark.parametrize('node_list', [VALID_LINEAR_CLF_PIPELINE['statistical_clf']])
def test_clf(node_list, data_list=None):
    if data_list is None:
        data_list = UNI_MULTI_CLF
    result = [AbstractPipeline(task=TASK).evaluate_pipeline(node_list, data) for data in data_list]
    assert result is not None
