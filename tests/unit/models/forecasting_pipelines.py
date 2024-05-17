import numpy as np
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot_ind.core.architecture.pipelines.abstract_pipeline import AbstractPipeline
from fedot_ind.core.metrics.metrics_implementation import RMSE
from fedot_ind.core.repository.constanst_repository import VALID_LINEAR_TSF_PIPELINE
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels


BENCHMARK = 'M4'
TASK = 'ts_forecasting'
TASK_PARAMS = {'forecast_length': 14}
DATASET_NAME = 'D2600'


def test_stl_arima_tsf(node_list=VALID_LINEAR_TSF_PIPELINE['stl_arima']):
    result_dict = AbstractPipeline(
        task=TASK, task_params=TASK_PARAMS).evaluate_pipeline(
        node_list, DATASET_NAME)
    assert result_dict is not None


def test_lagged_lgbm(node_list=VALID_LINEAR_TSF_PIPELINE['topological_lgbm']):
    result_dict = AbstractPipeline(
        task=TASK, task_params=TASK_PARAMS).evaluate_pipeline(
        node_list, DATASET_NAME)
    assert result_dict is not None


def test_topological_tsf(
        node_list=VALID_LINEAR_TSF_PIPELINE['topological_lgbm']):
    result_dict = AbstractPipeline(
        task=TASK, task_params=TASK_PARAMS).evaluate_pipeline(
        node_list, 'Q2124')
    assert result_dict is not None


def test_ar(node_list=VALID_LINEAR_TSF_PIPELINE['ar']):
    result_dict = AbstractPipeline(
        task=TASK, task_params=TASK_PARAMS).evaluate_pipeline(
        node_list, DATASET_NAME)
    assert result_dict is not None


def test_smoothing_ar(node_list=VALID_LINEAR_TSF_PIPELINE['smoothed_ar']):
    result_dict = AbstractPipeline(
        task=TASK, task_params=TASK_PARAMS).evaluate_pipeline(
        node_list, DATASET_NAME)
    assert result_dict is not None


def test_gaussian_ar(node_list=VALID_LINEAR_TSF_PIPELINE['gaussian_ar']):
    result_dict = AbstractPipeline(
        task=TASK, task_params=TASK_PARAMS).evaluate_pipeline(
        node_list, DATASET_NAME)
    assert result_dict is not None


def test_eigen_autoregression(
        node_list=VALID_LINEAR_TSF_PIPELINE['eigen_autoregression']):
    result_dict = AbstractPipeline(
        task=TASK, task_params=TASK_PARAMS).evaluate_pipeline(
        node_list, DATASET_NAME)
    assert result_dict is not None


def test_glm_tsf(node_list=VALID_LINEAR_TSF_PIPELINE['glm']):
    result_dict = AbstractPipeline(
        task=TASK, task_params=TASK_PARAMS).evaluate_pipeline(
        node_list, DATASET_NAME)
    assert result_dict is not None


def test_nbeats_tsf(node_list=VALID_LINEAR_TSF_PIPELINE['nbeats']):
    result_dict = AbstractPipeline(
        task=TASK, task_params=TASK_PARAMS).evaluate_pipeline(
        node_list, DATASET_NAME)
    assert result_dict is not None


def test_composite_tsf_pipeline(node_list=VALID_LINEAR_TSF_PIPELINE['nbeats']):
    result_dict = AbstractPipeline(
        task=TASK, task_params=TASK_PARAMS).evaluate_pipeline(
        node_list, DATASET_NAME)
    assert result_dict is not None
