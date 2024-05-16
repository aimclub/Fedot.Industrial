from fedot_ind.core.repository.constanst_repository import VALID_LINEAR_REG_PIPELINE
from fedot_ind.core.architecture.pipelines.abstract_pipeline import AbstractPipeline

MULTI_REG = ['AppliancesEnergy']
TASK = 'regression'

def test_quantile_lgbm_reg(node_list=VALID_LINEAR_REG_PIPELINE['statistical_lgbmreg'], data_list=None):
    if data_list is None:
        data_list = MULTI_REG
    result = [AbstractPipeline(task=TASK).evaluate_pipeline(node_list, data) for data in data_list][0]
    assert result is not None


def test_recurrence_reg(node_list=VALID_LINEAR_REG_PIPELINE['recurrence_reg'], data_list=None):
    if data_list is None:
        data_list = MULTI_REG
    result = [AbstractPipeline(task=TASK).evaluate_pipeline(node_list, data) for data in data_list][0]
    assert result is not None


def test_wavelet_reg(node_list=VALID_LINEAR_REG_PIPELINE['wavelet_statistical_reg'], data_list=None):
    if data_list is None:
        data_list = MULTI_REG
    result = [AbstractPipeline(task=TASK).evaluate_pipeline(node_list, data) for data in data_list][0]
    assert result is not None


def test_fourier_reg(node_list=VALID_LINEAR_REG_PIPELINE['fourier_statistical_reg'], data_list=None):
    if data_list is None:
        data_list = MULTI_REG
    result = [AbstractPipeline(task=TASK).evaluate_pipeline(node_list, data) for data in data_list][0]
    assert result is not None


def test_eigen_reg(node_list=VALID_LINEAR_REG_PIPELINE['eigen_statistical_reg'], data_list=None):
    if data_list is None:
        data_list = MULTI_REG
    result = [AbstractPipeline(task=TASK).evaluate_pipeline(node_list, data) for data in data_list][0]
    assert result is not None


def test_channel_filtration_reg(node_list=VALID_LINEAR_REG_PIPELINE['channel_filtration_statistical_reg'],
                                data_list=None):
    if data_list is None:
        data_list = MULTI_REG
    result = [AbstractPipeline(task=TASK).evaluate_pipeline(node_list, data) for data in data_list][0]
    assert result is not None


def test_composite_clf_pipeline(node_list=VALID_LINEAR_REG_PIPELINE['composite_reg'], data_list=None):
    if data_list is None:
        data_list = MULTI_REG
    result = [AbstractPipeline(task=TASK).evaluate_pipeline(node_list, data) for data in data_list][0]
    assert result is not None

