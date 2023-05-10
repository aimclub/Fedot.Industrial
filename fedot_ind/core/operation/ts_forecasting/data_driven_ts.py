from typing import TypeVar, Optional

import matplotlib.pyplot as plt
import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from pymonad.either import Either
from pymonad.list import ListMonad

from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.operation.transformation.regularization.spectrum import singular_value_hard_threshold

class_type = TypeVar("T", bound="DataDrivenBasis")


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
        self.n_components = params.get('n_components', 3)
        self.window_size = params.get('window_size')
        self.basis = None

    def predict(self, input_data: InputData) -> OutputData:
        trajectory_transformer = HankelMatrix(time_series=input_data.features, window_size=self.window_size)
        data = trajectory_transformer.trajectory_matrix
        self.ts_length = trajectory_transformer.ts_length
        self.U, self.Sigma, self.VT = self._get_basis(data)
        pipeline = PipelineBuilder().add_node('ar').build()
        predict_per_component = []
        for row in self.VT:
            train_data = InputData(idx=np.arange(row.shape[0]), features=row, target=row,
                                   data_type=input_data.data_type,
                                   task=input_data.task)
            test_data = InputData(idx=input_data.idx, features=row, target=input_data.target,
                                  data_type=input_data.data_type,
                                  task=input_data.task)
            pipeline.fit(train_data)
            predict = np.ravel(pipeline.predict(test_data).predict)
            predict_per_component.append(predict)
        predict_per_component = np.array(predict_per_component)
        reconstructed_target = self.reconstruct_basis(predict_per_component, ts_length=predict_per_component.shape[1])
        return reconstructed_target.sum(axis=1)

    def fit(self, input_data: InputData):
        pass

    def _get_1d_basis(self, data):
        svd = lambda x: ListMonad(np.linalg.svd(x))
        threshold = lambda Monoid: ListMonad([Monoid[0],
                                              singular_value_hard_threshold(singular_values=Monoid[1],
                                                                            beta=data.shape[0] / data.shape[1],
                                                                            threshold=None),
                                              Monoid[2]]) if self.n_components is None else ListMonad([Monoid[0][:, :self.n_components],
                                                                                                       Monoid[1][
                                                                                                       :self.n_components],
                                                                                                       Monoid[2][:self.n_components]])

        U, sigma, VT = Either.insert(data).then(svd).then(threshold).value[0]
        return U, sigma, VT

    def _get_basis(self, data):
        basis = self._get_1d_basis(data)
        return basis

    def reconstruct_basis(self, new_V, ts_length):
        if len(self.Sigma.shape) > 1:
            multi_reconstruction = lambda x: self.reconstruct_basis(U=self.U, Sigma=x, VT=new_V, ts_length=ts_length)
            TS_comps = list(map(multi_reconstruction, self.Sigma))
        else:
            rank = self.Sigma.shape[0]
            TS_comps = np.zeros((ts_length, rank))
            for i in range(rank):
                X_elem = self.Sigma[i] * np.outer(self.U[:, i], new_V[i, :])
                X_rev = X_elem[::-1]
                eigenvector = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]
                TS_comps[:, i] = eigenvector[-ts_length:]
        return TS_comps
