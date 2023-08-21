from typing import TypeVar, Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from golem.core.tuning.simultaneous import SimultaneousTuner
from pymonad.either import Either
from sklearn.metrics import mean_absolute_percentage_error

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
        self.estimator = params.get('estimator', PipelineBuilder().add_node('ar').build())
        self.SV_threshold = None

        self.decomposer = None
        self.basis = None

    def predict(self, input_data: InputData) -> OutputData:
        trajectory_transformer = HankelMatrix(time_series=input_data.features, window_size=self.window_size)
        data = trajectory_transformer.trajectory_matrix
        self.decomposer = SpectrumDecomposer(data, trajectory_transformer.ts_length)

        self.window_size = trajectory_transformer.window_length
        self.SV_threshold = self.estimate_singular_values(data)
        self.decomposer = SpectrumDecomposer(data, trajectory_transformer.ts_length, self.SV_threshold)
        basis = self.get_svd(data).T
        precise_of_approximation = mean_absolute_percentage_error(input_data.features, np.sum(basis, axis=0))
        if precise_of_approximation > 0.3:
            print(precise_of_approximation, 'use remain')
            np.append(basis, input_data.features - np.sum(basis, axis=0))
        reconstructed_target = []
        for i, row in enumerate(basis):
            # auto_model = Fedot(problem='ts_forecasting', task_params=input_data.task.task_params,
            #                    timeout=0.5,
            #                    n_jobs=1)

            train_data = InputData(idx=np.arange(row.shape[0]), features=row, target=row,
                                   data_type=input_data.data_type,
                                   task=input_data.task)
            test_data = InputData(idx=input_data.idx, features=row, target=None,
                                  data_type=input_data.data_type,
                                  task=input_data.task)
            pipeline_tuner = TunerBuilder(train_data.task) \
                .with_tuner(SimultaneousTuner) \
                .with_metric(RegressionMetricsEnum.MAE) \
                .with_iterations(5) \
                .with_cv_folds(2) \
                .with_validation_blocks(1) \
                .build(train_data)
            self.estimator = pipeline_tuner.tune(self.estimator)
            self.estimator.fit(train_data)
            predict = np.ravel(self.estimator.predict(test_data).predict)
            reconstructed_target.append(predict)
            self.estimator.unfit()
        return np.array(reconstructed_target).sum(axis=0)

    def fit(self, input_data: InputData):
        pass

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
        components = Either.insert(data).then(self.decomposer.svd).then(self.decomposer.threshold).then(
            self.decomposer.data_driven_basis).value[0]

        return components
