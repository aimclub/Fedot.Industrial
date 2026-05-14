from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import Optional

import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.metrics_repository import RegressionMetricsEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum
from golem.core.tuning.simultaneous import SimultaneousTuner

from fedot_ind.core.repository.industrial_implementations.data_transformation import prepare_lagged_table_data
from fedot_ind.core.tuning.search_space import get_industrial_search_space


def resolve_lagged_window_size(time_series_length: int, window_size_percent: float) -> int:
    candidate = int(round(float(time_series_length) * 0.01 * float(window_size_percent)))
    return int(max(2, min(candidate, max(2, int(time_series_length) - 1))))


class LaggedAR:
    """Compatibility wrapper for the historical lagged forecasting shell.

    The public API intentionally matches the legacy `LaggedAR` contract, while the
    implementation now routes forecasting through the `hankelisation -> channel_model`
    pipeline introduced in the refactored forecasting stack.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        self.params = params if isinstance(params, OperationParameters) else OperationParameters(**dict(params or {}))
        self.channel_model = str(self.params.get('channel_model', 'ridge'))
        self.window_size = self.params.get('window_size', 10)
        self.window_size_percent = self.params.get('window_size_percent')
        self.stride = int(self.params.get('stride', 1))
        self.custom_search_space = None
        self.replace_default_search_space = True
        self.tuning_params = {
            'metric': RegressionMetricsEnum.RMSE,
            'tuner': SimultaneousTuner,
            'cv_folds': None,
            'n_jobs': 1,
            'tuning_iterations': 10,
        }
        self.tuned_model = None
        self.resolved_window_size_ = None
        self.resolved_hankel_stride_ = None
        self.ts_patch_len = None

    def _resolve_window_size(self, input_data: InputData) -> int:
        series_length = int(np.asarray(input_data.features).shape[0])
        if self.window_size_percent is not None:
            return resolve_lagged_window_size(series_length, float(self.window_size_percent))
        return resolve_lagged_window_size(series_length, float(self.window_size))

    def _resolve_hankel_stride(self) -> int:
        return int(max(1, self.stride))

    def _define_forecasting_pipeline_model(self):
        return (
            PipelineBuilder()
            .add_node('hankelisation', params={
                'window_size': int(self.resolved_window_size_),
                'stride': int(self.resolved_hankel_stride_),
            })
            .add_node(self.channel_model)
            .build()
        )

    def _create_pcd(self, input_data: InputData, is_fit_stage: bool):
        return prepare_lagged_table_data(
            input_data,
            window_size=int(self.resolved_window_size_),
            stride=int(self.resolved_hankel_stride_),
            is_fit_stage=bool(is_fit_stage),
        )

    def _define_tuning_data(self, train_data: InputData):
        tuning_data = deepcopy(train_data)
        tuning_data.data_type = DataTypesEnum.table
        tuning_data.task.task_type = TaskTypesEnum.regression
        return tuning_data

    def _is_industrial_repository_active(self) -> bool:
        return False

    @contextmanager
    def _industrial_repository_scope(self):
        from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

        if self._is_industrial_repository_active():
            yield
            return

        initializer = IndustrialModels()
        initializer.setup_repository()
        try:
            yield
        finally:
            initializer.setup_default_repository()

    def _build_forecasting_tuner(self, model_to_tune, tuning_params, train_data):
        search_space = self._define_search_space()
        pipeline_tuner = (
            TunerBuilder(train_data.task)
            .with_search_space(search_space)
            .with_tuner(tuning_params["tuner"])
            .with_cv_folds(tuning_params.get("cv_folds", None))
            .with_n_jobs(tuning_params.get("n_jobs", 1))
            .with_metric(tuning_params["metric"])
            .with_iterations(tuning_params.get("tuning_iterations", 20))
            .build(train_data)
        )
        model_to_tune = pipeline_tuner.tune(model_to_tune)
        model_to_tune.fit(train_data)
        del pipeline_tuner
        return model_to_tune

    def build_tuner(self, model_to_tune, tuning_params, train_data):
        custom_search_space = get_industrial_search_space(self)
        search_space = PipelineSearchSpace(
            custom_search_space=custom_search_space,
            replace_default_search_space=True,
        )
        pipeline_tuner = (
            TunerBuilder(train_data.task)
            .with_search_space(search_space)
            .with_tuner(tuning_params['tuner'])
            .with_cv_folds(tuning_params.get('cv_folds', None))
            .with_n_jobs(tuning_params.get('n_jobs', 1))
            .with_metric(tuning_params['metric'])
            .with_iterations(tuning_params.get('tuning_iterations', 50))
            .build(self._define_tuning_data(train_data))
        )
        model_to_tune = pipeline_tuner.tune(model_to_tune)
        model_to_tune.fit(train_data)
        return model_to_tune

    def _build_forecasting_tuner(self, model_to_tune, tuning_params, train_data):
        custom_search_space = get_industrial_search_space(self)
        search_space = PipelineSearchSpace(
            custom_search_space=custom_search_space,
            replace_default_search_space=True,
        )
        pipeline_tuner = (
            TunerBuilder(train_data.task)
            .with_search_space(search_space)
            .with_tuner(tuning_params['tuner'])
            .with_cv_folds(tuning_params.get('cv_folds', None))
            .with_n_jobs(tuning_params.get('n_jobs', 1))
            .with_metric(tuning_params['metric'])
            .with_iterations(tuning_params.get('tuning_iterations', 50))
            .build(train_data)
        )
        model_to_tune = pipeline_tuner.tune(model_to_tune)
        model_to_tune.fit(train_data)
        return model_to_tune

    def _fit_hankel_pipeline(self, input_data: InputData):
        with self._industrial_repository_scope():
            model_to_tune = self._define_forecasting_pipeline_model()
            self.tuned_model = self._build_forecasting_tuner(
                model_to_tune=model_to_tune,
                tuning_params=self.tuning_params,
                train_data=input_data,
            )
        return self

    def fit(self, input_data: InputData):
        self.resolved_window_size_ = self._resolve_window_size(input_data)
        self.resolved_hankel_stride_ = self._resolve_hankel_stride()
        self.ts_patch_len = self.resolved_window_size_
        return self._fit_hankel_pipeline(input_data)

    def predict(self, input_data: InputData) -> OutputData:
        if self.tuned_model is None:
            raise ValueError('LaggedAR is not fitted.')
        return self.tuned_model.predict(input_data)

    def predict_for_fit(self, input_data: InputData):
        return self.predict(input_data)
