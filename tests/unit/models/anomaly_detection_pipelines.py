from fedot_ind.core.architecture.pipelines.abstract_pipeline import AbstractPipeline
from fedot_ind.core.repository.constanst_repository import VALID_LINEAR_DETECTION_PIPELINE

UNI_MULTI_CLF = [dict(benchmark='valve1',
                      dataset='1')]
TASK = 'classification'
TASK_PARAMS = dict(industrial_strategy='anomaly_detection',
                   detection_window=10)


def test_isolation_forest(
        node_list=VALID_LINEAR_DETECTION_PIPELINE['iforest_detector'],
        data_list=None):
    if data_list is None:
        data_list = UNI_MULTI_CLF
    result = [
        AbstractPipeline(
            task=TASK,
            task_params=TASK_PARAMS).evaluate_pipeline(
            node_list,
            data) for data in data_list]
    assert result is not None


def test_arima_fault(
        node_list=VALID_LINEAR_DETECTION_PIPELINE['arima_detector'],
        data_list=None):
    if data_list is None:
        data_list = UNI_MULTI_CLF
    result = [
        AbstractPipeline(
            task=TASK,
            task_params=TASK_PARAMS).evaluate_pipeline(
            node_list,
            data) for data in data_list]
    assert result is not None


def test_encoder_detector(
        node_list=VALID_LINEAR_DETECTION_PIPELINE['conv_ae_detector'],
        data_list=None):
    if data_list is None:
        data_list = UNI_MULTI_CLF
    result = [
        AbstractPipeline(
            task=TASK,
            task_params=TASK_PARAMS).evaluate_pipeline(
            node_list,
            data) for data in data_list]
    assert result is not None
