from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder


class EigenAR(ModelImplementation):
    """ Generalized linear models implementation """

    def __init__(self, params: OperationParameters):
        super().__init__(params)

    # @use_industrial_fedot_client
    def _create_forecasting_model(self, data_fold_features, task):
        train_fold = InputData.from_numpy_time_series(features_array=data_fold_features,
                                                      target_array=data_fold_features,
                                                      task=task)
        composite_pipeline = PipelineBuilder().add_node('fedot_forecast').build()
        composite_pipeline.fit(train_fold)
        model = composite_pipeline.root_node.fitted_operation.model
        return model, train_fold

    def _eigen_ensemble(self, decomposed_time_series: OutputData):
        node_dict = {}
        for i in range(decomposed_time_series.predict.shape[1]):
            data_fold_features = decomposed_time_series.predict[:, i, :].squeeze()
            model, train_fold = self._create_forecasting_model(data_fold_features, decomposed_time_series.task)
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
