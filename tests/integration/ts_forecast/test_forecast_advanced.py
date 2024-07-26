import pytest

from itertools import product
from tests.integration.integration_test_utils import data, launch_api

import pandas as pd
from pathlib import Path
import random
import numpy as np

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot_ind.api.main import FedotIndustrial
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot_ind.api.main import FedotIndustrial
from fedot_ind.api.utils.path_lib import PROJECT_PATH

from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate

FINETUNE = False
TASK = 'ts_forecasting'
FORECAST_LENGTH = 8

# def test_forecasting():
#     dataset_name = {'benchmark': 'M4',
#                     'dataset': 'D3257',
#                     'task_params': {'forecast_length': FORECAST_LENGTH}}
#     initial_assumptions = {
#         'nbeats': PipelineBuilder().add_node('nbeats_model'),
#         # 'industrial': PipelineBuilder().add_node(
#         #     'eigen_basis',
#         #     params={
#         #         'low_rank_approximation': False,
#         #         'rank_regularization': 'explained_dispersion'}).add_node('ar')
#     }
#     for assumption in initial_assumptions.keys():
#         api_config = dict(problem=TASK,
#                           metric='rmse',
#                           timeout=0.05,
#                           task_params={'forecast_length': FORECAST_LENGTH},
#                           n_jobs=-1,
#                           industrial_strategy='forecasting_assumptions',
#                           initial_assumption=initial_assumptions[assumption],
#                           logging_level=20)
#         result_dict = ApiTemplate(api_config=api_config,
#                                   metric_list=('rmse',)
#                                   ).eval(dataset=dataset_name, finetune=finetune)
#         current_metric = result_dict['metrics']
#         assert current_metric is not None
        
        
def test_forecasting_exogenous():
    dataset_name = PROJECT_PATH + \
        '/data/m4/datasets/Daily-train.csv'
    train_data = pd.read_csv(dataset_name, usecols=['V2', 'V3', 'V4', 'V5'])
    exog_var = ['V2', 'V3', 'V4']
    exog_ts = train_data[exog_var].values
    ts = train_data['V5'].values
    target = ts[-FORECAST_LENGTH:].flatten()
    input_data = (ts, target)
    initial_assumptions = {
        'nbeats': PipelineBuilder().add_node('nbeats_model'),
        # 'industrial': PipelineBuilder().add_node(
        #     'eigen_basis',
        #     params={
        #         'low_rank_approximation': False,
        #         'rank_regularization': 'explained_dispersion'}).add_node('ar')
    }

    for assumption in initial_assumptions.keys():
        api_config = dict(problem=TASK,
                          metric='rmse',
                          timeout=0.05,
                          task_params={'forecast_length': FORECAST_LENGTH, 'data_type': 'time_series', 
                                       'supplementary_data': {'feature_name': exog_var}},
                          n_jobs=-1,
                          industrial_strategy='forecasting_exogenous',
                          initial_assumption=initial_assumptions[assumption],
                          industrial_strategy_params={'exog_variable': exog_ts, 'data_type': 'time_series', 
                                                      'supplementary_data': {'feature_name': exog_var}},
                          logging_level=0)
        
        industrial = FedotIndustrial(**api_config)
        industrial.fit(input_data)
        assert not np.isnan(industrial.predict(input_data)).any()
