import json
from contextlib import contextmanager
from copy import deepcopy
from typing import Optional

import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.metrics_repository import RegressionMetricsEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum
from golem.core.tuning.simultaneous import SimultaneousTuner

from fedot_ind.core.models.ts_forecasting.stage_tuning import build_forecasting_stage_tuning_plan
from fedot_ind.core.repository.industrial_implementations.data_transformation import prepare_lagged_table_data
from fedot_ind.core.tuning.search_space import get_industrial_search_space
from fedot_ind.tools.serialisation.path_lib import PATH_TO_DEFAULT_PARAMS


def resolve_lagged_window_size(time_series_length: int, window_size_percent: float) -> int:
    resolved_window = round(time_series_length * 0.01 * window_size_percent)
    max_window = max(2, time_series_length - 1)
    return max(2, min(max_window, resolved_window))


class LaggedAR(ModelImplementation):
    """Implementation of a composite of a topological extractor and autoreg atomized models"""

    def __init__(self, params: Optional[OperationParameters] = None):
        params = params or OperationParameters()
        super().__init__(params)
        self.channel_model = self.params.get("channel_model", "ridge")
        self.window_size = self.params.get("window_size", 10)
        self.window_size_percent = self.window_size
        self.hankel_stride = int(self.params.get("stride", 1))
        self.resolved_window_size_ = None
        self.resolved_hankel_stride_ = None
        self.tuning_params = dict(
            tuner=SimultaneousTuner,
            metric=self.params.get("metric", RegressionMetricsEnum.RMSE),
            tuning_timeout=1,
            tuning_early_stop=10,
            tuning_iterations=20,
        )

    def _load_default_operation_params(self):
        with open(PATH_TO_DEFAULT_PARAMS) as json_data:
            self.default_operation_params = json.load(json_data)
        self.default_channel_model_params = self.default_operation_params[self.channel_model]

    def _define_model(self):
        self._load_default_operation_params()
        self.model = PipelineBuilder().add_node(self.channel_model,
                                                params=self.default_channel_model_params).build()
        return self.model

    def _define_forecasting_pipeline_model(self):
        self._load_default_operation_params()
        self.model = (
            PipelineBuilder()
            .add_node('hankelisation',
                      params={'window_size': self.resolved_window_size_,
                              'stride': self.resolved_hankel_stride_})
            .add_node(self.channel_model, params=self.default_channel_model_params)
            .build()
        )
        return self.model

    def _define_search_space(self):
        custom_search_space = get_industrial_search_space(self)
        search_space = PipelineSearchSpace(
            custom_search_space=custom_search_space, replace_default_search_space=True
        )

        return search_space

    def _define_tuning_data(self, train_data):
        tuning_data = deepcopy(train_data)
        tuning_data.data_type = DataTypesEnum.table
        # After lagged/page construction the tuner optimizes a standard
        # multi-output regression model on tabular rows, not a raw
        # ts_forecasting problem. Keeping ts_forecasting here routes FEDOT
        # tuning through time-series validation/metric logic and breaks
        # multi-target Ridge tuning for targets shaped like (n_samples, horizon).
        tuning_data.task.task_type = TaskTypesEnum.regression
        return tuning_data

    def _create_tuner(self, search_space, tuning_params, tuning_data):
        return (
            TunerBuilder(tuning_data.task)
            .with_search_space(search_space)
            .with_tuner(tuning_params["tuner"])
            .with_n_jobs(1)
            .with_metric(tuning_params["metric"])
            .with_iterations(tuning_params.get("tuning_iterations", 20))
            .build(tuning_data)
        )

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
        tuning_data = self._define_tuning_data(train_data)
        search_space = self._define_search_space()
        pipeline_tuner = self._create_tuner(search_space, tuning_params, tuning_data)
        model_to_tune = pipeline_tuner.tune(model_to_tune)
        model_to_tune.fit(train_data)
        del pipeline_tuner
        return model_to_tune

    def _resolve_window_size(self, input_data: InputData) -> int:
        time_series_length = input_data.features.shape[0]
        return resolve_lagged_window_size(time_series_length, self.window_size_percent)

    def _resolve_hankel_stride(self) -> int:
        max_stride = max(1, self.resolved_window_size_ // 2)
        return max(1, min(max_stride, int(self.hankel_stride)))

    def _prepare_regression_core_data(self, input_data: InputData, is_fit_stage: bool = True):
        if self.resolved_window_size_ is None:
            raise ValueError("Lagged regression core can not be prepared before the window size is resolved.")
        return prepare_lagged_table_data(
            deepcopy(input_data),
            window_size=self.resolved_window_size_,
            stride=self.resolved_hankel_stride_ or 1,
            is_fit_stage=is_fit_stage,
        )

    def _create_pcd(self, input_data, is_fit_stage: bool = True):
        return self._prepare_regression_core_data(input_data, is_fit_stage)

    def _is_industrial_repository_active(self) -> bool:
        from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

        repository = IndustrialModels()
        data_repo = OperationTypesRepository.__repository_dict__.get('data_operation', {}).get('file')
        model_repo = OperationTypesRepository.__repository_dict__.get('model', {}).get('file')
        return (
                data_repo == repository.industrial_data_operation_path
                and model_repo == repository.industrial_model_path
        )

    @contextmanager
    def _industrial_repository_scope(self):
        from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

        repository = IndustrialModels()
        already_active = self._is_industrial_repository_active()
        if not already_active:
            repository.setup_repository()
        try:
            yield
        finally:
            if not already_active:
                repository.setup_default_repository()

    def _fit_hankel_pipeline(self, input_data: InputData):
        with self._industrial_repository_scope():
            model_to_tune = self._define_forecasting_pipeline_model()
            self.tuned_model = self._build_forecasting_tuner(
                model_to_tune=model_to_tune,
                tuning_params=self.tuning_params,
                train_data=input_data,
            )
        return self

    def fit(self, input_data):
        self.resolved_window_size_ = self._resolve_window_size(input_data)
        self.resolved_hankel_stride_ = self._resolve_hankel_stride()
        # Preserve the old attribute name for compatibility with downstream
        # code that still reads `ts_patch_len` directly.
        self.ts_patch_len = self.resolved_window_size_
        self._fit_hankel_pipeline(input_data)

        del self.tuning_params
        return self

    def _predict_from_hankel_pipeline(self, input_data: InputData) -> OutputData:
        with self._industrial_repository_scope():
            prediction = self.tuned_model.predict(input_data)
        forecast_length = input_data.task.task_params.forecast_length
        prediction.predict = np.ravel(prediction.predict)[-forecast_length:]
        return prediction

    def _predict(self, input_data: InputData) -> OutputData:
        return self._predict_from_hankel_pipeline(input_data)

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        return self._predict(input_data)

    def predict(self, input_data: InputData) -> OutputData:
        return self._predict(input_data)

    def get_stage_tuning_plan(self) -> dict[str, object]:
        return build_forecasting_stage_tuning_plan(
            'lagged_forecaster',
            {
                'window_size': self.window_size,
                'stride': self.hankel_stride,
                'channel_model': self.channel_model,
            },
        ).to_dict()
