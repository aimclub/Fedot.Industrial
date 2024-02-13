from copy import deepcopy

import pandas as pd
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot_ind.core.architecture.preprocessing.data_convertor import FedotConverter
from fedot_ind.core.operation.transformation.basis.eigen_basis import EigenBasisImplementation
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.operation.transformation.regularization.spectrum import reconstruct_basis

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

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.window_size_method = params.get('window_size_method')
        self.history_lookback = params.get('history_lookback', 100)
        self.low_rank_approximation = params.get('low_rank_approximation', False)
        self._decomposer = None
        self._rank_thr = None
        self._window_size = None
        self.horizon = None
        self.preprocess_to_lagged = False

    def predict(self, input_data: InputData) -> OutputData:
        hankel_matrix = HankelMatrix(time_series=input_data.features,
                                     window_size=self._decomposer.window_size).trajectory_matrix
        U, s, VT = np.linalg.svd(hankel_matrix)
        n_components = list(range(self._rank_thr))
        PCT = np.concatenate([U[:, 0].reshape(1, -1), np.array([np.sum([U[:, i], U[:, i + 1]], axis=0)
                                                                for i in n_components if i != 0 and i % 2 != 0])]).T
        current_dynamics = np.concatenate([VT[0, :].reshape(1, -1), np.array([np.sum([VT[i, :], VT[i + 1, :]], axis=0)
                                                                              for i in n_components if
                                                                              i != 0 and i % 2 != 0])])
        forecast_by_channel, model_by_channel = self._predict_channel(input_data,
                                                                      current_dynamics,
                                                                      self.horizon)

        forecasted_dynamics = np.concatenate([current_dynamics,
                                              np.vstack(list(forecast_by_channel.values()))], axis=1)
        basis = reconstruct_basis(U=PCT,
                                  Sigma=s[:PCT.shape[1]],
                                  VT=forecasted_dynamics,
                                  ts_length=input_data.features.shape[0] + self.horizon)

        summed_basis = np.array(basis).sum(axis=1)
        reconstructed_forecast = summed_basis[-self.horizon:]
        reconstructed_features = summed_basis[:-self.horizon]

        #error = input_data.features.ravel() - reconstructed_features
        prediction = reconstructed_forecast
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
        basis = self._decomposer.transform(ts)
        self._rank_thr = basis.predict.shape[0]
        components_correlation = np.concatenate([basis.predict[0, :].reshape(1, -1), np.array([np.sum(
            [basis.predict[i, :], basis.predict[i + 1, :]], axis=0) for i in range(basis.predict.shape[0]) if i != 0
                                                                                                              and i % 2 != 0])])

        reconstructed_features = np.array(components_correlation).sum(axis=0)
        return reconstructed_features

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        if self.horizon is None:
            self.horizon = input_data.task.task_params.forecast_length
        input_data.features = input_data.features[-self.history_lookback:].squeeze()
        self._decomposer = EigenBasisImplementation({'low_rank_approximation': self.low_rank_approximation,
                                                     'rank_regularization': 'explained_dispersion'})
        predict = self.__predict_for_fit(input_data)
        return predict

    def _predict_channel(self, input_data: InputData, component_dynamics, forecast_length: int):
        comp = deepcopy(input_data)
        comp.features = component_dynamics
        comp.idx = np.arange(component_dynamics.shape[1])
        ts_channels = comp.features
        forecast_by_channel = {}
        model_by_channel = {}

        model = PipelineBuilder().add_node('gaussian_filter').add_node('ar')
        # model = PipelineBuilder().add_node('ar')
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
