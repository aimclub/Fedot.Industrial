from typing import Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
import tensorly as tl
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from joblib import delayed, Parallel
from pymonad.either import Either
from pymonad.list import ListMonad
from scipy import stats
from scipy.spatial.distance import cdist
from tensorly.decomposition import parafac
from tqdm import tqdm

from fedot_ind.core.architecture.preprocessing.data_convertor import NumpyConverter
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
        self.tensor_approximation = params.get('tensor_approximation', False)
        self.logging_params.update({'WS': self.window_size})
        self.explained_dispersion = []
        self.SV_threshold = None
        self.svd_estimator = RSVDDecomposition()

    def __repr__(self):
        return 'EigenBasisImplementation'

    def _transform(self, input_data: InputData) -> np.array:
        """Method for transforming all samples

        """
        if isinstance(input_data, InputData):
            features = input_data.features
        else:
            features = np.array(ListMonad(*input_data.values.tolist()).value)
            features = np.array([series[~np.isnan(series)] for series in features])
        features = NumpyConverter(data=features).convert_to_torch_format()
        if self.tensor_approximation:
            predict = self._get_multidim_basis(features)
        else:
            if self.SV_threshold is None:
                self.SV_threshold = max(self.get_threshold(data=features), 2)
                self.logging_params.update({'SV_thr': self.SV_threshold})
            predict = []
            for dimension in range(features.shape[1]):
                parallel = Parallel(n_jobs=self.n_processes, verbose=0, pre_dispatch="2*n_jobs")
                v = parallel(delayed(self._transform_one_sample)(sample) for sample in features[:, dimension, :])
                predict.append(np.array(v) if len(v) > 1 else v[0])

        self.predict = np.concatenate(predict, axis=1)

        if input_data.task.task_params is None:
            input_data.task.task_params = self.__repr__()
        elif input_data.task.task_params not in [self.__repr__(), 'LargeFeatureSpace']:
            input_data.task.task_params.feature_filter = self.__repr__()

        predict = OutputData(idx=input_data.idx,
                             features=input_data.features,
                             predict=self.predict,
                             task=input_data.task,
                             target=input_data.target,
                             data_type=DataTypesEnum.table,
                             supplementary_data=input_data.supplementary_data)
        return predict

    def get_threshold(self, data) -> int:
        svd_numbers = []
        with tqdm(total=len(data), desc='SVD estimation') as pbar:
            for dimension in range(data.shape[1]):
                dimension_rank = []
                for signal in data[:, dimension, :]:
                    dimension_rank.append(self._transform_one_sample(signal, svd_flag=True))
                    pbar.update(1)
                svd_numbers.append(stats.mode(dimension_rank).mode)
        try:
            return stats.mode(svd_numbers).mode[0]
        except Exception:
            return stats.mode(svd_numbers).mode

    def _transform_one_sample(self, series: np.array, svd_flag: bool = False):
        trajectory_transformer = HankelMatrix(time_series=series, window_size=self.window_size)
        data = trajectory_transformer.trajectory_matrix
        self.ts_length = trajectory_transformer.ts_length
        rank = self.estimate_singular_values(data)
        if svd_flag:
            return rank
        else:
            return self._get_1d_basis(data)

    def estimate_singular_values(self, data):
        svd = lambda x: ListMonad(self.svd_estimator.rsvd(tensor=x, approximation=self.low_rank_approximation))
        basis = Either.insert(data).then(svd).value[0]
        spectrum = [s_val for s_val in basis[1] if s_val > 0.001]
        rank = len(spectrum)
        self.explained_dispersion.append([round(x / sum(spectrum) * 100) for x in spectrum][:rank])
        # self.left_approx_sv, self.right_approx_sv = basis[0], basis[2]
        return rank

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
        rank = round(data.shape[2] / 10)
        beta = data.shape[2] / data.shape[0]

        tensor_decomposition = lambda x: ListMonad(parafac(tl.tensor(x), rank=rank).factors)
        multi_threshold = lambda x: singular_value_hard_threshold(singular_values=x,
                                                                  beta=beta,
                                                                  threshold=None)

        threshold = lambda Monoid: ListMonad([Monoid[0],
                                              list(map(multi_threshold, Monoid[1])),
                                              Monoid[2].T])
        data_driven_basis = lambda Monoid: ListMonad(reconstruct_basis(Monoid[0],
                                                                       Monoid[1],
                                                                       Monoid[2],
                                                                       ts_length=data.shape[2]))

        basis = np.array(
            Either.insert(data).then(tensor_decomposition).then(threshold).then(data_driven_basis).value[0])

        basis = basis.reshape(basis.shape[1], -1)

        return basis

