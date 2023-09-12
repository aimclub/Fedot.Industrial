from typing import TypeVar, Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from pymonad.either import Either
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error
from statsforecast.arima import AutoARIMA
from statsforecast.models import AutoTheta, AutoETS

from fedot_ind.core.operation.decomposition.SpectrumDecomposition import SpectrumDecomposer
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix

class_type = TypeVar("T", bound="DataDrivenBasis")

import numpy as np

try:
    import seaborn
except:
    pass
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 11, 4


def MASE(A, F, y_train):
    return mean_absolute_scaled_error(A, F, y_train=y_train)


class DataDrivenForForecastingBasisImplementation(ModelImplementation):
    """DataDriven basis
        Example:
            ts1 = np.random.rand(200)
            ts2 = np.random.rand(200)
            ts = [ts1, ts2]
            bss = DataDrivenBasisImplementation({'n_components': 3, 'window_size': 30})
            basis_multi = bss._transform(ts)
            basis_1d = bss._transform(ts1)
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.window_size = params.get('window_size')
        self.SV_threshold = None

        self.decomposer = None
        self.basis = None

    def predict(self, input_data: InputData) -> OutputData:
        forecast_length = input_data.task.task_params.forecast_length
        trajectory_transformer = HankelMatrix(time_series=input_data.features, window_size=self.window_size)
        data = trajectory_transformer.trajectory_matrix

        self.window_size = trajectory_transformer.window_length
        self.decomposer = SpectrumDecomposer(data,
                                             trajectory_transformer.ts_length + forecast_length,
                                             self.SV_threshold)
        U, s, VT = self.get_svd(data)

        new_VT = []
        for i in range(s.shape[0]):
            row = VT[i]
            model_theta = AutoTheta()
            model_arima = AutoARIMA()
            model_ets = AutoETS()
            forecast = []
            for model in [model_theta, model_arima, model_ets]:
                model.fit(row)
                p = model.predict(forecast_length)['mean']
                forecast.append(p)
            pred = np.median(np.array(forecast), axis=0)
            new_VT.append(np.concatenate([row, pred]))

        new_VT = np.array(new_VT)
        basis = self.reconstruct_basis(U, s, new_VT).T

        self.decomposer = SpectrumDecomposer(data,
                                             trajectory_transformer.ts_length,
                                             self.SV_threshold)
        self.train_basis = self.reconstruct_basis(U, s, VT).T

        return np.array(basis).sum(axis=0)[-forecast_length:]

    def fit(self, input_data: InputData):
        pass

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        trajectory_transformer = HankelMatrix(time_series=input_data.features, window_size=self.window_size)
        data = trajectory_transformer.trajectory_matrix
        self.decomposer = SpectrumDecomposer(data, trajectory_transformer.ts_length)

        self.window_size = trajectory_transformer.window_length
        self.SV_threshold = self.estimate_singular_values(data)
        self.decomposer = SpectrumDecomposer(data,
                                             trajectory_transformer.ts_length,
                                             self.SV_threshold)
        U, s, VT = self.get_svd(data)
        basis = self.reconstruct_basis(U, s, VT).T
        reconstructed_features = np.array(basis).sum(axis=0)
        return reconstructed_features

    def estimate_singular_values(self, data):
        basis = Either.insert(data).then(self.decomposer.svd).value[0]
        spectrum = [s_val for s_val in basis[1] if s_val > 0.001]
        # self.left_approx_sv, self.right_approx_sv = basis[0], basis[2]
        return len(spectrum)

    def get_combined_components(self, U, Sigma, V_T):
        components = Either.insert([U, Sigma, V_T]).then(
            self.decomposer.data_driven_basis).value[0]
        return components

    def get_svd(self, data):
        components = Either.insert(data).then(self.decomposer.svd).then(self.decomposer.threshold).value[0]
        return components

    def reconstruct_basis(self, U, s, VT):
        return Either.insert([U, s, VT]).then(self.decomposer.data_driven_basis).value[0]
