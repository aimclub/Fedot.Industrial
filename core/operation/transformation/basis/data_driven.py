from typing import Tuple, TypeVar

import numpy as np
import tensorly as tl
from pymonad.either import Either
from pymonad.list import ListMonad
from tensorly.decomposition import parafac

from core.operation.transformation.basis.abstract_basis import BasisDecompositionImplementation
from core.operation.transformation.data.hankel import HankelMatrix
from core.operation.transformation.regularization.spectrum import singular_value_hard_threshold, reconstruct_basis

class_type = TypeVar("T", bound="DataDrivenBasis")


class DataDrivenBasisImplementation(BasisDecompositionImplementation):
    """DataDriven basis
    """

    def _transform(self, features: ListMonad):
        trajectory_transformer = HankelMatrix(time_series=features)
        self.data = trajectory_transformer.trajectory_matrix
        self.ts_length = trajectory_transformer.ts_length
        return self._get_basis()

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

        self.basis = Either.insert(data).then(svd).then(threshold).then(data_driven_basis).value[0]

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
        res = []
        if type(self.data) == list:
            for arr in self.data:
                res.append(self._get_1d_basis(arr))
        self.basis = np.array(res)
        return self.basis



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
