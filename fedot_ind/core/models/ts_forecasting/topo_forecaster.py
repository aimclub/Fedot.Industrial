from copy import deepcopy
from typing import Optional

import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import \
    LaggedTransformationImplementation
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from golem.core.tuning.optuna_tuner import OptunaTuner

from fedot_ind.core.tuning.search_space import get_industrial_search_space


class TopologicalAR(ModelImplementation):
    """Implementation of a composite of a topological extractor and autoreg atomized models"""

    def __init__(self, params: Optional[OperationParameters] = None):
        params = params or OperationParameters()
        super().__init__(params)
        self.channel_model = self.params.get("channel_model", "ar")
        self.fitted_model_dict = None
        self.lagged_node = LaggedTransformationImplementation(OperationParameters(window_size=10))
        self.topo_ts = None
        self.tuning_params = dict(
            tuner=OptunaTuner,
            metric=self.params.get("metric", RegressionMetricsEnum.RMSE),
            tuning_timeout=5,
            tuning_early_stop=50,
            tuning_iterations=100,
        )

    def build_tuner(self, model_to_tune, tuning_params, train_data):
        def _create_tuner(tuning_params, tuning_data):
            custom_search_space = get_industrial_search_space(self)
            search_space = PipelineSearchSpace(
                custom_search_space=custom_search_space, replace_default_search_space=True
            )

            return (
                TunerBuilder(train_data.task)
                .with_search_space(search_space)
                .with_tuner(tuning_params["tuner"])
                .with_n_jobs(-1)
                .with_metric(tuning_params["metric"])
                .with_timeout(tuning_params.get("tuning_timeout", 5))
                .with_iterations(tuning_params.get("tuning_iterations", 100))
                .with_early_stopping_rounds(tuning_params.get("tuning_early_stop", 50))
                .build(tuning_data)
            )

        pipeline_tuner = _create_tuner(tuning_params, train_data)
        model_to_tune = pipeline_tuner.tune(model_to_tune, False)
        model_to_tune.fit(train_data)
        del pipeline_tuner
        return model_to_tune

    def fit(self, input_data):
        new_input_data = self._convert_input_data(deepcopy(input_data))
        self.topo_ts = PipelineBuilder().add_node("topological_features").build()

        # TODO: adapt this topo_ts pipeline to retrieve a point cloud from lagged matrix
        # TODO: change topological_features to Industrial topo implementation call
        point_cloud = self.topo_ts.fit(new_input_data).predict.squeeze()
        self.fitted_model_dict = {}
        for component_idx, ts_component in enumerate(point_cloud):
            copy_input_data = deepcopy(new_input_data)
            copy_input_data.features = ts_component
            tuned_model = self.build_tuner(
                model_to_tune=PipelineBuilder().add_node(self.channel_model).build(),
                tuning_params=self.tuning_params,
                train_data=copy_input_data,
            )
            self.fitted_model_dict.update({component_idx: tuned_model})

        del self.tuning_params
        return self

    def _predict(self, input_data: InputData) -> OutputData:
        new_input_data = self._convert_input_data(deepcopy(input_data))
        point_cloud = self.topo_ts.predict(new_input_data).predict.reshape(
            len(self.fitted_model_dict), new_input_data.features.shape[0]
        )
        prediction = []
        for component_idx, ts_component in enumerate(point_cloud):
            copy_input_data = deepcopy(new_input_data)
            copy_input_data.features = ts_component
            prediction.append(self.fitted_model_dict[component_idx].predict(copy_input_data).predict)
        prediction = np.stack(prediction)
        output_data = self._convert_to_output(
            input_data, predict=np.sum(prediction, axis=0), data_type=DataTypesEnum.table
        )
        return output_data

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        return self._predict(input_data)

    def predict(self, input_data: InputData) -> OutputData:
        return self._predict(input_data)

    def _convert_input_data(self, input_data: InputData) -> InputData:
        if len(input_data.target.shape) < 2:
            return self.lagged_node.transform_for_fit(input_data)
        return input_data


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    features = np.random.rand(100)
    target = np.random.rand(100)
    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=20))
    data = InputData(idx=np.arange(100),
                     features=features,
                     target=target,
                     task=task,
                     data_type=DataTypesEnum.ts)
    train_data, test_data = train_test_data_setup(data)

    topo_ar = TopologicalAR()
    topo_ar.fit(train_data)
    predictions = topo_ar.predict(test_data).predict

    plt.plot(train_data.idx, train_data.target, label='train_data')
    plt.plot(test_data.idx, test_data.target, label='test_data')
    plt.plot(test_data.idx, predictions, label='sample_prediction')
    plt.legend()
    plt.show()
