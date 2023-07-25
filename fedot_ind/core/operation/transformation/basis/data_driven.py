import math
from multiprocessing import Pool
from typing import Optional, Tuple, TypeVar

import numpy as np
import tensorly as tl
from fedot.core.operations.operation_parameters import OperationParameters
from pymonad.either import Either
from pymonad.list import ListMonad
from tensorly.decomposition import parafac
from tqdm import tqdm

from fedot_ind.core.architecture.preprocessing import InputData
from fedot_ind.core.operation.decomposition.matrix_decomposition.fast_svd import bksvd
from fedot_ind.core.operation.transformation.basis.abstract_basis import BasisDecompositionImplementation
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.operation.transformation.regularization.spectrum import reconstruct_basis, \
    singular_value_hard_threshold

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
        self.window_size = params.get('window_size')
        self.basis = None
        self.SV_threshold = None
        self.sv_selector = params.get('sv_selector')

        self.logging_params.update({'WS': self.window_size,
                                    'SV_selector': self.sv_selector,
                                    })

    def _transform(self, input_data: InputData) -> np.array:
        """Method for transforming all samples

        """
        features = np.array(ListMonad(*input_data.features.tolist()).value)
        features = np.array([series[~np.isnan(series)] for series in features])

        if self.SV_threshold is None:
            self.SV_threshold = self.get_threshold(data=features,
                                                   selector=self.sv_selector)
            self.logging_params.update({'SV_thr': self.SV_threshold})

        with Pool(self.n_processes) as p:
            v = list(tqdm(p.imap(self._transform_one_sample, features),
                          total=features.shape[0],
                          desc=f'{self.__class__.__name__} transform',
                          postfix=f'{self.logging_params}',
                          colour='red',
                          unit='ts',
                          ascii=False,
                          position=0,
                          initial=0,
                          leave=True,
                          )
                     )
        predict = np.array(v)
        return predict

    def get_threshold(self, data, selector: str):

        selectors = {'median': np.median,
                     '0.75%': lambda x: np.quantile(x, 0.75),
                     '0.25%': lambda x: np.quantile(x, 0.25)}

        svd_numbers = []
        with tqdm(total=len(data), desc='SVD estimation') as pbar:
            for signal in data:
                svd_numbers.append(self._transform_one_sample(signal, svd_flag=True))
                pbar.update(1)

        return math.ceil(selectors[selector](svd_numbers))

    def _transform_one_sample(self, series: np.array, svd_flag: bool = False):
        trajectory_transformer = HankelMatrix(time_series=series, window_size=self.window_size)
        data = trajectory_transformer.trajectory_matrix
        self.ts_length = trajectory_transformer.ts_length
        if svd_flag:
            return self.estimate_singular_values(data)
        return self._get_basis(data)

    def estimate_singular_values(self, data):
        svd = lambda x: ListMonad(bksvd(tensor=x))
        basis = Either.insert(data).then(svd).value[0]
        spectrum = [s_val for s_val in basis[1] if s_val > 0.001]
        #self.left_approx_sv, self.right_approx_sv = basis[0], basis[2]
        return len(spectrum)

    def _get_1d_basis(self, data):
        data_driven_basis = lambda Monoid: ListMonad(reconstruct_basis(Monoid[0],
                                                                       Monoid[1],
                                                                       Monoid[2],
                                                                       ts_length=self.ts_length))
        threshold = lambda Monoid: ListMonad([Monoid[0],
                                              Monoid[1][:self.SV_threshold],
                                              Monoid[2]])
        svd = lambda x: ListMonad(bksvd(tensor=x))
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
