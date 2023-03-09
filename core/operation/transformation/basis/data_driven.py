import numpy as np
from typing import Tuple, TypeVar
from pymonad.list import ListMonad
from core.operation.transformation.basis.abstract_basis import BasisDecomposition
from core.operation.transformation.data.hankel import HankelMatrix
from core.operation.transformation.regularization.spectrum import singular_value_hard_threshold, reconstruct_basis
from pymonad.either import Either
import tensorly as tl
from tensorly.decomposition import tucker, parafac, non_negative_tucker

class_type = TypeVar("T", bound="DataDrivenBasis")


class DataDrivenBasis(BasisDecomposition):
    """DataDriven basis
    """

    def _get_1d_basis(self):
        svd = lambda x: ListMonad(np.linalg.svd(x))
        threshold = lambda Monoid: ListMonad([Monoid[0],
                                              singular_value_hard_threshold(singular_values=Monoid[1],
                                                                            beta=self.data.shape[0] / self.data.shape[
                                                                                1],
                                                                            threshold=None),
                                              Monoid[2]]) if self.n_components is None else ListMonad([Monoid[0],
                                                                                                       Monoid[1][
                                                                                                       :self.n_components],
                                                                                                       Monoid[2]])
        data_driven_basis = lambda Monoid: ListMonad(reconstruct_basis(Monoid[0],
                                                                       Monoid[1],
                                                                       Monoid[2],
                                                                       ts_length=self.ts_length))

        self.basis = Either.insert(self.data).then(svd).then(threshold).then(data_driven_basis).value[0]

        return self.basis

    def _get_multidim_basis(self):
        rank = round(self.data[0].shape[0] / 10)
        beta = self.data[0].shape[0] / self.data[0].shape[1]

        tensor_decomposition = lambda x: ListMonad(parafac(tl.tensor(x), rank=rank).factors)
        multi_threshold = lambda x: singular_value_hard_threshold(singular_values=x,
                                                                  beta=beta,
                                                                  threshold=None)

        threshold = lambda Monoid: ListMonad([Monoid[1],
                                              list(map(multi_threshold, Monoid[0])),
                                              Monoid[2].T]) if self.n_components is None else ListMonad([Monoid[1],
                                                                                                         Monoid[0][
                                                                                                         :self.n_components],
                                                                                                         Monoid[2].T])
        data_driven_basis = lambda Monoid: ListMonad(reconstruct_basis(Monoid[0],
                                                                       Monoid[1],
                                                                       Monoid[2],
                                                                       ts_length=self.ts_length))

        self.basis = Either.insert(self.data).then(tensor_decomposition).then(threshold).then(data_driven_basis).value[
            0]

        return self.basis

    def _get_basis(self):
        if type(self.data) == list:
            self.basis = self._get_multidim_basis()
        else:
            self.basis = self._get_1d_basis()

        if self.min_rank is None:
            self.min_rank = self.basis.shape[1]
        else:
            self.min_rank = min(self.basis.shape[1], self.min_rank)
        return self.basis

    def fit(self, data, **kwargs):
        if 'window_length' not in kwargs.keys():
            kwargs['window_length'] = None
        trajectory_transformer = HankelMatrix(time_series=data, window_length=kwargs['window_length'])
        self.data = trajectory_transformer.trajectory_matrix
        self.ts_length = trajectory_transformer.ts_length
        return self._get_basis()

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
