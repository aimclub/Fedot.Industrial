from copy import deepcopy

from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot_ind.core.architecture.preprocessing.data_convertor import NumpyConverter, FedotConverter

try:
    import seaborn
except ImportError:
    pass
from typing import Optional

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from pymonad.either import Either

from fedot_ind.core.operation.decomposition.spectrum_decomposition import SpectrumDecomposer
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.operation.transformation.window_selector import WindowSizeSelector


class SSAForecasterImplementation(ModelImplementation):
    """Model for forecasting univariate timeseries with Singular Spectrum Decomposition.
    For given time series ``T`` we construct trajectory matrix (hankel matrix) ``X``, where
    ``X = U x S x V_t``. After decomposition, we forecast ``V_t`` rows separately, and after
    that reconstruct basis. Note that we use only few components to reconstruct basis. Other
    components considered as error (we just sample them).

    Attributes:
        window_size_method: str, method for estimating window size for SSA forecaster

    Example:
        To use this operation you can create pipeline as follows::

            from fedot_ind.core.architecture.settings.computational import backend_methods as np
            from fedot.core.data.data import InputData
            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from examples.example_utils import get_ts_data
            from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

            forecast_length = 13
            train_data, test_data, dataset_name = get_ts_data('m4_monthly', forecast_length)
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('ssa_forecaster').build()
                pipeline.fit(train_data)
                prediction = pipeline.predict(test_data)
                print(prediction)

    """
    LAST_VALUES_THRESHOLD = 100

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

        self.window_size_method = params.get('window_size_method')
        self.n_processes = 1
        self._SV_threshold = None
        self._decomposer = None
        self._window_size = None
        self.horizon = None
        self.preprocess_to_lagged = False

    def __preprocess_for_fedot(self, features):
        features = np.squeeze(features)

        if len(features.shape) == 1:
            self.preprocess_to_lagged = False
        else:
            if features.shape[1] != 1:
                self.preprocess_to_lagged = True
                self._window_size = features.shape[1]
                ts_length = features.shape[1] + \
                            features.shape[0] - 1
                ts_length = features.shape[1]
                self._decomposer = SpectrumDecomposer(features,
                                                      ts_length)

        if self.preprocess_to_lagged:
            self.seq_len = features.shape[0] + \
                           features.shape[1]
        else:
            self.seq_len = features.shape[0]
            features = features[-self.LAST_VALUES_THRESHOLD:]
            trajectory_transformer = HankelMatrix(
                time_series=features, window_size=self._window_size)
            features = trajectory_transformer.trajectory_matrix
            self._decomposer = SpectrumDecomposer(features,
                                                  trajectory_transformer.ts_length)

        return features

    def predict(self, input_data: InputData) -> OutputData:
        features = input_data.features
        trajectory_matrix = self.__preprocess_for_fedot(features)
        U, s, VT = self._predict_loop(trajectory_matrix)
        s_basis = s[:self._decomposer.thr]
        current_dynamics = VT[:self._decomposer.thr]
        forecast_by_channel, model_by_channel = self._predict_channel(input_data,
                                                                      current_dynamics,
                                                                      self.horizon)

        forecasted_dynamics = np.concatenate([current_dynamics,
                                              np.vstack(list(forecast_by_channel.values()))], axis=1)
        self._decomposer.ts_length = self._decomposer.ts_length + self.horizon
        basis = self.reconstruct_basis(U, s_basis, forecasted_dynamics).T

        summed_basis = np.array(basis).sum(axis=0)
        reconstructed_forecast = summed_basis[-self.horizon:]
        reconstructed_features = summed_basis[:-self.horizon]

        error = input_data.features[-self.horizon:] - \
                reconstructed_features[-self.horizon:]
        prediction = reconstructed_forecast + error
        predict_data = FedotConverter(input_data).convert_to_output_data(prediction=prediction,
                                                                         predict_data=input_data,
                                                                         output_data_type=input_data.data_type)
        return predict_data

    def _predict_loop(self, input_data: InputData) -> tuple:
        U, s, VT = self.get_svd(input_data)
        return U, s, VT

    def fit(self, input_data: InputData):
        pass

    def __predict_for_fit(self, ts):
        U, s, VT = self._predict_loop(ts)
        basis = self.reconstruct_basis(U, s, VT).T
        reconstructed_features = np.array(basis).sum(axis=0)
        return reconstructed_features

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        features = input_data.features
        self.target = input_data.target
        self.task_type = input_data.task
        if self.horizon is None:
            self.horizon = input_data.task.task_params.forecast_length

        data = self.__preprocess_for_fedot(features)
        if self.preprocess_to_lagged:
            predict = [self.__predict_for_fit(ts.reshape(1, -1)) for ts in data]
            predict = np.array(predict)
        else:
            predict = self.__predict_for_fit(data)
        return predict

    def _predict_channel(self, input_data: InputData, component_dynamics, forecast_length: int):

        comp = deepcopy(input_data)
        comp.features = component_dynamics
        comp.idx = np.arange(component_dynamics.shape[1])
        ts_channels = comp.features
        forecast_by_channel = {}
        model_by_channel = {}

        model = PipelineBuilder().add_node('gaussian_filter').add_node('ar')
        #model = PipelineBuilder().add_node('ar')
        for index, ts_comp in enumerate(ts_channels):
            comp.features = ts_comp
            component_model = model.build()
            component_model.fit(comp)
            forecast = component_model.predict(comp, forecast_length)
            forecast_by_channel.update({f'{index}_channel': forecast.predict})
            model_by_channel.update({f'{index}_channel': component_model})

        return forecast_by_channel, model_by_channel

    def get_svd(self, data):
        components = Either.insert(data).then(self._decomposer.svd).then(
            self._decomposer.threshold).value[0]
        return components

    def reconstruct_basis(self, U, s, VT):
        return Either.insert([U, s, VT]).then(self._decomposer.data_driven_basis).value[0]
