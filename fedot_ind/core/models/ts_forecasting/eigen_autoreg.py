from copy import deepcopy

import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.metrics_repository import RegressionMetricsEnum
from golem.core.tuning.optuna_tuner import OptunaTuner

from fedot_ind.core.tuning.search_space import get_industrial_search_space


class EigenAR(ModelImplementation):
    """ Generalized linear models implementation """

    def __init__(self, params: OperationParameters):
        super().__init__(params)
        self.periodicity_length = [5, 7, 14, 29, 30, 31]
        self.channel_model = params.get('channel_model', 'ar')
        self.tuning_params = dict(
            tuner=OptunaTuner,
            metric=RegressionMetricsEnum.RMSE,
            tuning_timeout=2,
            tuning_early_stop=50,
            tuning_iterations=100)

    def build_tuner(self, model_to_tune, tuning_params, train_data):
        def _create_tuner(tuning_params, tuning_data):
            custom_search_space = get_industrial_search_space(self)
            search_space = PipelineSearchSpace(custom_search_space=custom_search_space,
                                               replace_default_search_space=True)
            pipeline_tuner = TunerBuilder(
                train_data.task).with_search_space(search_space).with_tuner(
                tuning_params['tuner']).with_n_jobs(-1).with_metric(
                tuning_params['metric']).with_timeout(
                tuning_params.get('tuning_timeout', 15.0)).with_iterations(
                tuning_params.get('tuning_iterations', 100)).with_early_stopping_rounds(
                tuning_params.get('tuning_early_stop', 50)).build(tuning_data)

            return pipeline_tuner

        pipeline_tuner = _create_tuner(tuning_params, train_data)
        model_to_tune = pipeline_tuner.tune(model_to_tune, False)
        model_to_tune.fit(train_data)
        del pipeline_tuner
        return model_to_tune

    def fit(self, input_data):
        self.eigen_ts = PipelineBuilder().add_node('eigen_basis', params={
            'low_rank_approximation': False,
            'rank_regularization': 'explained_dispersion'}).build()
        decomposed_time_series = self.eigen_ts.fit(input_data).predict.squeeze()
        self.fitted_model_dict = {}
        for component_idx, ts_component in enumerate(decomposed_time_series):
            copy_input_data = deepcopy(input_data)
            copy_input_data.features = ts_component
            tuned_model = self.build_tuner(model_to_tune=PipelineBuilder().add_node(self.channel_model).build(),
                                           tuning_params=self.tuning_params,
                                           train_data=copy_input_data)
            self.fitted_model_dict.update({component_idx: tuned_model})

        return self

    def _predict(self, input_data):
        decomposed_time_series = self.eigen_ts.predict(input_data).predict.reshape(len(self.fitted_model_dict),
                                                                                   input_data.features.shape[0])
        prediction = []
        for component_idx, ts_component in enumerate(decomposed_time_series):
            copy_input_data = deepcopy(input_data)
            copy_input_data.features = ts_component
            prediction.append(self.fitted_model_dict[component_idx].predict(copy_input_data).predict)
        prediction = np.stack(prediction)
        output_data = self._convert_to_output(input_data,
                                              predict=np.sum(prediction, axis=0),
                                              data_type=DataTypesEnum.table)
        return output_data

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        return self._predict(input_data)

    def predict(self, input_data: InputData) -> OutputData:
        return self._predict(input_data)
