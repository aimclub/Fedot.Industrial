from typing import TypeVar, Optional

from fedot.api.main import Fedot
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import TsForecastingParams
from golem.core.tuning.simultaneous import SimultaneousTuner
from matplotlib import pyplot as plt
from pymonad.either import Either

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
        estimator_name = params.get('estimator', 'ar')
        print(estimator_name)
        ESTIMATORS = {
            'arima': PipelineBuilder().add_node('arima').build(),
            'ridge': PipelineBuilder().add_node('lagged').add_node('ridge').build(),
            'ar': PipelineBuilder().add_node('ar').build(),
            'ets': PipelineBuilder().add_node('ets').build()}
        self.estimator = ESTIMATORS[estimator_name]

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
            train_data = InputData(idx=np.arange(row.shape[0]), features=row, target=row,
                                   data_type=input_data.data_type,
                                   task=input_data.task)
            test_data = InputData(idx=input_data.idx, features=row, target=None,
                                  data_type=input_data.data_type,
                                  task=input_data.task)
            pipeline_tuner = TunerBuilder(train_data.task) \
                .with_tuner(SimultaneousTuner) \
                .with_metric(RegressionMetricsEnum.MAE) \
                .with_iterations(3) \
                .with_cv_folds(None) \
                .with_validation_blocks(1) \
                .build(train_data)
            self.estimator = pipeline_tuner.tune(self.estimator)
            self.estimator.fit(train_data)
            predict = np.ravel(self.estimator.predict(test_data).predict)
            self.estimator.unfit()
            # plt.plot(U[i])
            # plt.plot(np.arange(len(row), len(row)+len(predict)), predict)
            new_VT.append(np.concatenate([row, predict]))

        new_VT = np.array(new_VT)
        basis = self.reconstruct_basis(U, s, new_VT).T

        self.decomposer = SpectrumDecomposer(data,
                                             trajectory_transformer.ts_length,
                                             self.SV_threshold)
        self.train_basis = self.reconstruct_basis(U, s, VT).T

        # plt.grid()
        # plt.show()
        # plt.plot(input_data.features)
        # for i in basis:
        #     plt.plot(i)
        #
        # plt.plot(np.arange(len(input_data.features), len(input_data.features) + forecast_length), np.array(basis).sum(axis=0)[-forecast_length:])
        # plt.show()
        reconstructed = np.array(basis).sum(axis=0)
        remains = input_data.features - reconstructed[:-forecast_length]

        train_data = InputData(idx=np.arange(remains.shape[0]), features=remains, target=remains,
                               data_type=input_data.data_type,
                               task=input_data.task)
        test_data = InputData(idx=input_data.idx, features=remains, target=None,
                              data_type=input_data.data_type,
                              task=input_data.task)
        fedot = Fedot(problem='ts_forecasting',
                      task_params=TsForecastingParams(forecast_length=forecast_length),
                      timeout=0.5)
        fedot.fit(train_data)
        fedot_predict = fedot.predict(test_data)

        predict = reconstructed[-forecast_length:] + fedot_predict
        return predict

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
        return np.array(basis).sum(axis=0)

    def estimate_singular_values(self, data):
        basis = Either.insert(data).then(self.decomposer.svd).value[0]
        spectrum = [s_val for s_val in basis[1] if s_val > 0.001]
        # self.left_approx_sv, self.right_approx_sv = basis[0], basis[2]
        return len(spectrum)

    def get_combined_components(self, U, Sigma, V_T):
        components = Either.insert([U, Sigma, V_T]).then(
            self.decomposer.data_driven_basis).then(
            self.decomposer.combine_components).value[0]

        return components

    def get_svd(self, data):
        components = Either.insert(data).then(self.decomposer.svd).then(self.decomposer.threshold).value[0]

        return components

    def reconstruct_basis(self, U, s, VT):
        return Either.insert([U, s, VT]).then(self.decomposer.data_driven_basis).value[0]
