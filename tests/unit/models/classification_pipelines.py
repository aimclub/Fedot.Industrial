from fedot_ind.core.architecture.pipelines.abstract_pipeline import AbstractPipeline
from fedot_ind.core.repository.constanst_repository import VALID_LINEAR_CLF_PIPELINE

UNI_MULTI_CLF = ['Earthquakes', 'ERing']
TASK = 'classification'


def test_quantile_clf(node_list=VALID_LINEAR_CLF_PIPELINE['statistical_clf'], data_list=None):
    if data_list is None:
        data_list = UNI_MULTI_CLF
    result = [AbstractPipeline(task=TASK).evaluate_pipeline(node_list, data) for data in data_list]
    assert result is not None


def test_quantile_lgbm_clf(node_list=VALID_LINEAR_CLF_PIPELINE['statistical_lgbm'], data_list=None):
    if data_list is None:
        data_list = UNI_MULTI_CLF
    result = [AbstractPipeline(task=TASK).evaluate_pipeline(node_list, data) for data in data_list]
    assert result is not None


def test_riemann_clf(node_list=VALID_LINEAR_CLF_PIPELINE['riemann_clf'], data_list=None):
    if data_list is None:
        data_list = UNI_MULTI_CLF
    result = [AbstractPipeline(task=TASK).evaluate_pipeline(node_list, data) for data in data_list]
    assert result is not None


def test_recurrence_clf(node_list=VALID_LINEAR_CLF_PIPELINE['recurrence_clf'], data_list=None):
    if data_list is None:
        data_list = UNI_MULTI_CLF
    result = [AbstractPipeline(task=TASK).evaluate_pipeline(node_list, data) for data in data_list]
    assert result is not None


def test_wavelet_clf(node_list=VALID_LINEAR_CLF_PIPELINE['wavelet_statistical'], data_list=None):
    if data_list is None:
        data_list = UNI_MULTI_CLF
    result = [AbstractPipeline(task=TASK).evaluate_pipeline(node_list, data) for data in data_list]
    assert result is not None


def test_fourier_clf(node_list=VALID_LINEAR_CLF_PIPELINE['fourier_statistical'], data_list=None):
    if data_list is None:
        data_list = UNI_MULTI_CLF
    result = [AbstractPipeline(task=TASK).evaluate_pipeline(node_list, data) for data in data_list]
    assert result is not None


def test_eigen_clf(node_list=VALID_LINEAR_CLF_PIPELINE['eigen_statistical'], data_list=None):
    if data_list is None:
        data_list = UNI_MULTI_CLF
    result = [AbstractPipeline(task=TASK).evaluate_pipeline(node_list, data) for data in data_list]
    assert result is not None


def test_channel_filtration_clf(node_list=VALID_LINEAR_CLF_PIPELINE['channel_filtration_statistical'], data_list=None):
    if data_list is None:
        data_list = UNI_MULTI_CLF
    result = [AbstractPipeline(task=TASK).evaluate_pipeline(node_list, data) for data in data_list]
    assert result is not None


def test_composite_clf_pipeline(node_list=VALID_LINEAR_CLF_PIPELINE['composite_clf'], data_list=None):
    if data_list is None:
        data_list = UNI_MULTI_CLF
    result = [AbstractPipeline(task=TASK).evaluate_pipeline(node_list, data) for data in data_list]
    assert result is not None

