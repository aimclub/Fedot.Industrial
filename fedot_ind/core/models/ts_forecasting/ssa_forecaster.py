try:
    import seaborn
except:
    pass
import math
from multiprocessing import cpu_count
from typing import TypeVar, Optional

import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from joblib import Parallel, delayed
from matplotlib.pylab import rcParams
from pymonad.either import Either
from statsforecast.arima import AutoARIMA

from fedot_ind.core.operation.decomposition.spectrum_decomposition import SpectrumDecomposer
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.operation.transformation.regularization.spectrum import sv_to_explained_variance_ratio
from fedot_ind.core.operation.transformation.window_selector import WindowSizeSelector

class_type = TypeVar("T", bound="DataDrivenBasis")
rcParams['figure.figsize'] = 11, 4


class SSAForecasterImplementation(ModelImplementation):
    """
    Model for forecasting uni-variate timeseries with Singular Spectrum Decomposition.
    For given time series T we construct trajectory matrix (hankel matrix) X.
    X = U x S x V_t. After decomposition, we forecast V_t rows separately and after it reconstruct basis. Note that we
    use only few components to reconstruct basis. Other components considered as error (we just sample them).

    Attributes:
        self.window_size_method: str, method for estimating window size for SSA forecaster

    Example:
        To use this operation you can create pipeline as follows::
            import numpy as np
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
        self.n_processes = math.ceil(cpu_count() * 0.7) if cpu_count() > 1 else 1
        self._SV_threshold = None
        self._decomposer = None
        self._window_size = None

    def predict(self, input_data: InputData) -> OutputData:
        forecast_length = input_data.task.task_params.forecast_length
        features = input_data.features[-self.LAST_VALUES_THRESHOLD:]
        trajectory_transformer = HankelMatrix(time_series=features, window_size=self._window_size)
        data = trajectory_transformer.trajectory_matrix

        self._window_size = trajectory_transformer.window_length
        self._decomposer = SpectrumDecomposer(data,
                                              trajectory_transformer.ts_length + forecast_length)
        U, s, VT = self.get_svd(data)

        s_basis = s[:self._SV_threshold]

        parallel = Parallel(n_jobs=self.n_processes, verbose=0, pre_dispatch="2*n_jobs")
        new_VT = np.array(
            parallel(delayed(self._predict_component)(sample, forecast_length) for sample in VT[:s_basis.shape[0]]))
        basis = self.reconstruct_basis(U, s_basis, new_VT).T

        summed_basis = np.array(basis).sum(axis=0)
        reconstructed_forecast = summed_basis[-forecast_length:]
        reconstructed_features = summed_basis[:-forecast_length]
        error = features[-forecast_length:] - reconstructed_features[-forecast_length:]

        return reconstructed_forecast + error

    def fit(self, input_data: InputData):
        pass

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        features = input_data.features[-self.LAST_VALUES_THRESHOLD:]
        self._window_size = int(WindowSizeSelector(method=self.window_size_method).get_window_size(features) * len(
            features) / 100)
        trajectory_transformer = HankelMatrix(time_series=features, window_size=self._window_size)
        data = trajectory_transformer.trajectory_matrix
        self._decomposer = SpectrumDecomposer(data, trajectory_transformer.ts_length)

        self._window_size = trajectory_transformer.window_length
        self._decomposer = SpectrumDecomposer(data,
                                              trajectory_transformer.ts_length)
        U, s, VT = self.get_svd(data)
        basis = self.reconstruct_basis(U, s, VT).T
        reconstructed_features = np.array(basis).sum(axis=0)
        return reconstructed_features

    def _predict_component(self, comp: np.array, forecast_length: int):
        estimated_seasonal_length = WindowSizeSelector('hac').get_window_size(comp)
        model_arima = AutoARIMA(period=estimated_seasonal_length)
        forecast = []
        for model in [model_arima]:
            model.fit(comp)
            p = model.predict(forecast_length)['mean']
            forecast.append(p)
        forecast = np.mean(np.array(forecast), axis=0)
        return np.concatenate([comp, forecast])

    def get_svd(self, data):
        components = Either.insert(data).then(self._decomposer.svd).then(self._decomposer.threshold).value[0]
        return components

    def reconstruct_basis(self, U, s, VT):
        return Either.insert([U, s, VT]).then(self._decomposer.data_driven_basis).value[0]
