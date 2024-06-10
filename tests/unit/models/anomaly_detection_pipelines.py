import pytest

from fedot_ind.core.architecture.pipelines.abstract_pipeline import AbstractPipeline
from fedot_ind.core.repository.constanst_repository import VALID_LINEAR_DETECTION_PIPELINE

UNI_MULTI_CLF = [dict(benchmark='valve1', dataset='1')]


@pytest.mark.parametrize('node_list', VALID_LINEAR_DETECTION_PIPELINE.values())
def test_anomaly_detector(node_list, data_list=None):
    if data_list is None:
        data_list = UNI_MULTI_CLF

    result = [
        AbstractPipeline(
            task='classification',
            task_params=dict(industrial_strategy='anomaly_detection', detection_window=10)
        ).evaluate_pipeline(node_list, data) for data in data_list
    ]
    assert result is not None
