import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from statsmodels import api as sm


class EigenAR(ModelImplementation):
    """ Generalized linear models implementation """

    def __init__(self, params: OperationParameters):
        super().__init__(params)
        self.periodicity_length = [5, 7, 14, 29, 30, 31]

    def _create_forecasting_model(self,
                                  data_fold_features,
                                  task,
                                  periodicity):
        if periodicity:
            composite_pipeline = PipelineBuilder().add_node('locf').build()
        else:
            composite_pipeline = PipelineBuilder().add_node('fedot_forecast').build()
        train_fold = InputData.from_numpy_time_series(features_array=data_fold_features,
                                                      target_array=data_fold_features,
                                                      task=task)
        composite_pipeline.fit(train_fold)
        model = composite_pipeline.root_node.fitted_operation.model if not periodicity else composite_pipeline
        return model, train_fold

    def _check_component_periodicity(self, TS_comps):
        def _validate_periodicity(first_peak, second_peak):
            diff = second_peak - first_peak
            if first_peak > 0.8 and second_peak > 0.8 and diff in self.periodicity_length:
                return True
            else:
                return False

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
