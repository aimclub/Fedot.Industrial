try:
    import seaborn
except:
    pass
from matplotlib.pylab import rcParams

import numpy as np
import math
from multiprocessing import cpu_count
from typing import TypeVar, Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from joblib import Parallel, delayed
from pymonad.either import Either
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error
from statsforecast.arima import AutoARIMA
from statsforecast.models import AutoTheta, AutoETS


from fedot_ind.core.operation.decomposition.spectrum_decomposition import SpectrumDecomposer
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.operation.transformation.regularization.spectrum import sv_to_explained_variance_ratio
from fedot_ind.core.operation.transformation.window_selector import WindowSizeSelector

class_type = TypeVar("T", bound="DataDrivenBasis")
rcParams['figure.figsize'] = 11, 4


def MASE(A, F, y_train):
    return mean_absolute_scaled_error(A, F, y_train=y_train)


class DataDrivenForForecastingBasisImplementation(ModelImplementation):
    """DataDriven basis
        Example:
            ts1 = np.random.rand(200)
            ts2 = np.random.rand(200)
            ts = [ts1, ts2]
            bss = EigenBasisImplementation({'n_components': 3, 'window_size': 30})
            basis_multi = bss._transform(ts)
            basis_1d = bss._transform(ts1)
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

        self.window_size = params.get('window_size')
        self.SV_threshold = None
        self.n_processes = math.ceil(cpu_count() * 0.7) if cpu_count() > 1 else 1
        self.decomposer = None
        self.basis = None

    def predict(self, input_data: InputData) -> OutputData:
        forecast_length = input_data.task.task_params.forecast_length
        features = input_data.features
        if len(input_data.features > 250):
            features = input_data.features[-250:]
        trajectory_transformer = HankelMatrix(time_series=features, window_size=self.window_size)
        data = trajectory_transformer.trajectory_matrix

        self.window_size = trajectory_transformer.window_length
        self.decomposer = SpectrumDecomposer(data,
                                             trajectory_transformer.ts_length + forecast_length,
                                             self.SV_threshold)
        U, s, VT = self.get_svd(data)

        parallel = Parallel(n_jobs=self.n_processes, verbose=0, pre_dispatch="2*n_jobs")
        new_VT = parallel(delayed(self._predict_component)(sample, forecast_length) for sample in VT[:s.shape[0]])
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
        features = input_data.features
        if len(input_data.features > 250):
            features = input_data.features[-250:]
        self.window_size = int(WindowSizeSelector(method='highest_autocorrelation').get_window_size(features) * len(
            features) / 100)
        trajectory_transformer = HankelMatrix(time_series=features, window_size=self.window_size)
        data = trajectory_transformer.trajectory_matrix
        self.decomposer = SpectrumDecomposer(data, trajectory_transformer.ts_length)

        self.window_size = trajectory_transformer.window_length
        self.SV_threshold = self.estimate_singular_values(data)
        self.decomposer = SpectrumDecomposer(data,
                                             trajectory_transformer.ts_length)
        U, s, VT = self.get_svd(data)
        basis = self.reconstruct_basis(U, s, VT).T

        min_rank = 1
        previous_ratio = 0
        while min_rank < len(s):
            variance_ratio = sv_to_explained_variance_ratio(s, min_rank)[0]
            if variance_ratio - previous_ratio < 3 and variance_ratio > 95:
                print(variance_ratio)
                break

            min_rank += 1
            previous_ratio = variance_ratio
        reconstructed_features = np.array(basis[:min_rank, :]).sum(axis=0)
        self.SV_threshold = min_rank
        return reconstructed_features

    def estimate_singular_values(self, data):
        basis = Either.insert(data).then(self.decomposer.svd).value[0]
        spectrum = [s_val for s_val in basis[1] if s_val > 0.001]
        # self.left_approx_sv, self.right_approx_sv = basis[0], basis[2]
        return len(spectrum)

    def _predict_component(self, comp: np.array, forecast_length: int):

        estimated_seasonal_length = WindowSizeSelector().get_window_size(comp)
        model_arima = AutoARIMA(period=estimated_seasonal_length)
        forecast = []
        for model in [model_arima]:
            model.fit(comp)
            p = model.predict(forecast_length)['mean']
            forecast.append(p)
        forecast = np.mean(np.array(forecast), axis=0)
        return np.concatenate([comp, forecast])

    def get_svd(self, data):
        components = Either.insert(data).then(self.decomposer.svd).then(self.decomposer.threshold).value[0]
        return components

    def reconstruct_basis(self, U, s, VT):
        return Either.insert([U, s, VT]).then(self.decomposer.data_driven_basis).value[0]
