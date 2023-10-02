from typing import Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
import tensorly as tl
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from joblib import delayed, Parallel
from pymonad.either import Either
from pymonad.list import ListMonad
from scipy import stats
from scipy.spatial.distance import cdist
from tensorly.decomposition import parafac
from tqdm import tqdm

from fedot_ind.core.operation.decomposition.matrix_decomposition.power_iteration_decomposition import RSVDDecomposition
from fedot_ind.core.operation.transformation.basis.abstract_basis import BasisDecompositionImplementation
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.operation.transformation.regularization.spectrum import reconstruct_basis, \
    singular_value_hard_threshold

class_type = TypeVar("T", bound="DataDrivenBasis")


class EigenBasisImplementation(BasisDecompositionImplementation):
    """Eigen basis decomposition implementation
        Example:
            ts1 = np.random.rand(200)
            ts2 = np.random.rand(200)
            ts = [ts1, ts2]
            bss = EigenBasisImplementation({'window_size': 30})
            basis_multi = bss._transform(ts)
            basis_1d = bss._transform(ts1)
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.window_size = params.get('window_size', 20)
        self.low_rank_approximation = params.get('low_rank_approximation', True)
        self.basis = None
        self.SV_threshold = None
        self.svd_estimator = RSVDDecomposition()
        self.logging_params.update({'WS': self.window_size})

    def _combine_components(self, predict):
        count = 0
        grouped_v = []
        for df in predict:
            tmp = pd.DataFrame(df)
            ff = cdist(metric='cosine', XA=tmp.values, XB=tmp.values)
            if ff[-1, -2] < 0.5:
                count += 1
            tmp.iloc[-2, :] = tmp.iloc[-2,] + tmp.iloc[-1, :]
            tmp.drop(tmp.tail(1).index, inplace=True)
            grouped_v.append(tmp.values)

        if count / len(predict) > 0.35:
            self.SV_threshold = grouped_v[0].shape[0]
            self.logging_params.update({'SV_thr': self.SV_threshold})
            return np.array(grouped_v)
        else:
            return predict

    def _transform(self, input_data: InputData) -> np.array:
        """Method for transforming all samples

        """
        if isinstance(input_data, InputData):
            features = np.array(ListMonad(*input_data.features.tolist()).value)
        else:
            features = np.array(ListMonad(*input_data.values.tolist()).value)
        features = np.array([series[~np.isnan(series)] for series in features])

        if self.SV_threshold is None:
            self.SV_threshold = self.get_threshold(data=features)
            self.logging_params.update({'SV_thr': self.SV_threshold})

        parallel = Parallel(n_jobs=self.n_processes, verbose=0, pre_dispatch="2*n_jobs")
        v = parallel(delayed(self._transform_one_sample)(sample) for sample in features)
        predict = np.array(v)
        return predict

    def get_threshold(self, data) -> int:
        svd_numbers = []
        with tqdm(total=len(data), desc='SVD estimation') as pbar:
            for signal in data:
                svd_numbers.append(self._transform_one_sample(signal, svd_flag=True))
                pbar.update(1)
        return stats.mode(svd_numbers).mode[0]

    def _transform_one_sample(self, series: np.array, svd_flag: bool = False):
        trajectory_transformer = HankelMatrix(time_series=series, window_size=self.window_size)
        data = trajectory_transformer.trajectory_matrix
        self.ts_length = trajectory_transformer.ts_length
        if svd_flag:
            return self.estimate_singular_values(data)
        return self._get_basis(data)

    def estimate_singular_values(self, data):
        svd = lambda x: ListMonad(self.svd_estimator.rsvd(tensor=x, approximation=self.low_rank_approximation))
        basis = Either.insert(data).then(svd).value[0]
        spectrum = [s_val for s_val in basis[1] if s_val > 0.001]
        # self.left_approx_sv, self.right_approx_sv = basis[0], basis[2]
        return len(spectrum)

    def _get_1d_basis(self, data):
        data_driven_basis = lambda Monoid: ListMonad(reconstruct_basis(Monoid[0],
                                                                       Monoid[1],
                                                                       Monoid[2],
                                                                       ts_length=self.ts_length))
        threshold = lambda Monoid: ListMonad([Monoid[0],
                                              Monoid[1][:self.SV_threshold],
                                              Monoid[2]])
        svd = lambda x: ListMonad(self.svd_estimator.rsvd(tensor=x,
                                                          approximation=self.low_rank_approximation,
                                                          regularized_rank=self.SV_threshold))
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

    def evaluate_derivative(self: class_type,
                            coefs: np.array,
                            order: int = 1) -> Tuple[class_type, np.array]:
        basis = type(self)(
            domain_range=self.domain_range,
            n_basis=self.n_basis - order,
        )
        derivative_coefs = np.array([np.polyder(x[::-1], order)[::-1] for x in coefs])

        return basis, derivative_coefs
