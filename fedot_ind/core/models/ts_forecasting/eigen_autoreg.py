import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from pymonad.either import Either
from pymonad.maybe import Maybe
from statsmodels import api as sm


class EigenAR(ModelImplementation):
    """ Generalized linear models implementation """

    def __init__(self, params: OperationParameters):
        super().__init__(params)
        self.periodicity_length = [5, 7, 14, 29, 30, 31]
        self.channel_model = params.get('channel_model', 'ar')

    def _create_forecasting_model(self,
                                  data_fold_features,
                                  task,
                                  periodicity):
        train_fold = InputData.from_numpy_time_series(features_array=data_fold_features,
                                                      target_array=data_fold_features,
                                                      task=task)
        period = periodicity[1]
        weak_stationary_process = round(np.mean(data_fold_features)) == 0
        deterministic = True if weak_stationary_process else False
        build_model = lambda params: PipelineBuilder().add_node(**params).build()
        params_for_periodic = {'deterministic': deterministic,
                               'trend': 'n',
                               'seasonal': True,
                               'period': period}
        fedot_as_model = self.channel_model == 'fedot_forecast'

        model = Either(value=dict(operation_type=self.channel_model, params=params_for_periodic),
                       monoid=[dict(operation_type=self.channel_model, params={}), periodicity[0]]) \
            .either(left_function=build_model,
                    right_function=build_model)
        model = Maybe(value=model, monoid=[True, False]).then(function=lambda composite_pipeline:
        composite_pipeline.fit(train_fold) if fedot_as_model else composite_pipeline).value
        return model, train_fold

    def _check_component_periodicity(self, TS_comps):
        def _validate_periodicity(first_peak, second_peak):
            diff = second_peak - first_peak
            if first_peak > 0.8 and second_peak > 0.8 and diff in self.periodicity_length:
                return True, diff
            else:
                return False, diff

        acf = [sm.tsa.acf(TS_comps[:, i, :].flatten(), nlags=len(TS_comps[:, i, :].flatten()))
               for i in range(TS_comps.shape[1])]
        periodicity_dict = {}
        for idx, i in enumerate(acf):
            first_peak = np.argmax(i) + 1
            second_peak = np.argmax(i[first_peak:]) + first_peak + 1
            periodicity_dict.update({f'{idx}_component': _validate_periodicity(first_peak, second_peak)})
        return periodicity_dict

    def _eigen_ensemble(self, decomposed_time_series: OutputData):
        node_dict = {}
        periodicity_dict = self._check_component_periodicity(decomposed_time_series.predict)
        for i in range(decomposed_time_series.predict.shape[1]):
            data_fold_features = decomposed_time_series.predict[:, i, :].squeeze()
            model, train_fold = self._create_forecasting_model(data_fold_features,
                                                               decomposed_time_series.task,
                                                               periodicity_dict[f'{i}_component'])
            node_dict.update({f'{i}_component': dict(train_fold_data=train_fold,
                                                     composite_pipeline=model)})
        return node_dict

    def fit(self, input_data):
        eigen_ts = PipelineBuilder().add_node('eigen_basis', params={
            'low_rank_approximation': False,
            'rank_regularization': 'explained_dispersion'}).build()
        decomposed_time_series = eigen_ts.fit(input_data)
        model_dict = self._eigen_ensemble(decomposed_time_series)
        return model_dict

    def predict(self, input_data):
        return None

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        return None
