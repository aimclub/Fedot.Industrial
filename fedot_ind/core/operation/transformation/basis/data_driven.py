from typing import Tuple, TypeVar, Optional

import numpy as np
import tensorly as tl
from fedot.core.operations.operation_parameters import OperationParameters
from pymonad.either import Either
from pymonad.list import ListMonad
from tensorly.decomposition import parafac

from fedot_ind.core.operation.transformation.basis.abstract_basis import BasisDecompositionImplementation
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.operation.transformation.regularization.spectrum import singular_value_hard_threshold, reconstruct_basis

class_type = TypeVar("T", bound="DataDrivenBasis")


class DataDrivenBasisImplementation(BasisDecompositionImplementation):
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
        self.n_components = params.get('n_components')
        self.window_size = params.get('window_size')
        self.basis = None

    def _transform_one_sample(self, series: np.array):
        trajectory_transformer = HankelMatrix(time_series=series, window_size=self.window_size)
        data = trajectory_transformer.trajectory_matrix
        self.ts_length = trajectory_transformer.ts_length
        return self._get_basis(data)

    def _get_1d_basis(self, data):
        svd = lambda x: ListMonad(np.linalg.svd(x))
        threshold = lambda Monoid: ListMonad([Monoid[0],
                                              singular_value_hard_threshold(singular_values=Monoid[1],
                                                                            beta=data.shape[0] / data.shape[1],
                                                                            threshold=None),
                                              Monoid[2]]) if self.n_components is None else ListMonad([Monoid[0],
                                                                                                       Monoid[1][
                                                                                                       :self.n_components],
                                                                                                       Monoid[2]])
        data_driven_basis = lambda Monoid: ListMonad(reconstruct_basis(Monoid[0],
                                                                       Monoid[1],
                                                                       Monoid[2],
                                                                       ts_length=self.ts_length))

        basis = Either.insert(data).then(svd).then(threshold).then(data_driven_basis).value[0]

        return np.swapaxes(basis, 1, 0)

    def _get_multidim_basis(self, data):
        rank = round(data[0].shape[0] / 10)
        beta = data[0].shape[0] / data[0].shape[1]

        tensor_decomposition = lambda x: ListMonad(parafac(tl.tensor(x), rank=rank).factors)
        multi_threshold = lambda x: singular_value_hard_threshold(singular_values=x,
                                                                  beta=beta,
                                                                  threshold=None)

        threshold = lambda Monoid: ListMonad([Monoid[1],
                                              list(map(multi_threshold, Monoid[0])),
                                              Monoid[2].T]) if self.n_components is None else ListMonad([Monoid[1][
                                                                                                         :,
                                                                                                         :self.n_components],
                                                                                                         Monoid[0][
                                                                                                         :,
                                                                                                         :self.n_components],
                                                                                                         Monoid[2][
                                                                                                         :,
                                                                                                         :self.n_components].T])
        data_driven_basis = lambda Monoid: ListMonad(reconstruct_basis(Monoid[0],
                                                                       Monoid[1],
                                                                       Monoid[2],
                                                                       ts_length=self.ts_length))

        basis = np.array(
            Either.insert(data).then(tensor_decomposition).then(threshold).then(data_driven_basis).value[0])

        basis = basis.reshape(basis.shape[1], -1)

        return basis

    def evaluate_derivative(self:
                            class_type,
                            coefs: np.array,
                            order: int = 1) -> Tuple[class_type, np.array]:
        basis = type(self)(
            domain_range=self.domain_range,
            n_basis=self.n_basis - order,
        )
        derivative_coefs = np.array([np.polyder(x[::-1], order)[::-1] for x in coefs])

        return basis, derivative_coefs
