from typing import Optional, TypeVar

import numpy as np
import pandas as pd
import tensorly as tl
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from joblib import Parallel, delayed
from pymonad.either import Either
from pymonad.list import ListMonad
from scipy import stats
from scipy.spatial.distance import cdist
from tensorly.decomposition import parafac
from tqdm import tqdm

from fedot_ind.core.operation.decomposition.SpectrumDecomposition import SpectrumDecomposer
from fedot_ind.core.operation.implementation.basis.abstract_basis import BasisDecompositionImplementation
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.operation.transformation.regularization.spectrum import reconstruct_basis

class_type = TypeVar("T", bound="DataDrivenBasis")


class DataDrivenBasisImplementation(BasisDecompositionImplementation):
    """DataDriven basis
        Example:
            ts1 = np.random.rand(200)
            ts2 = np.random.rand(200)
            ts = [ts1, ts2]
            bss = DataDrivenBasisImplementation({'sv_selector': 'median', 'window_size': 30})
            basis_multi = bss._transform(ts)
            basis_1d = bss._transform(ts1)
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.window_size = params.get('window_size')
        self.basis = None
        self.SV_threshold = None
        # self.sv_selector = params.get('sv_selector')
        self.sv_selector = 'median'

        self.logging_params.update({'WS': self.window_size,
                                    'SV_selector': self.sv_selector,
                                    })

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
            self.SV_threshold = self.get_threshold(data=features,
                                                   selector=self.sv_selector)
            self.logging_params.update({'SV_thr': self.SV_threshold})

        parallel = Parallel(n_jobs=self.n_processes, verbose=0, pre_dispatch="2*n_jobs")
        v = parallel(delayed(self._transform_one_sample)(sample) for sample in features)
        predict = np.array(v)
        # new_shape = predict[0].shape[0]
        #
        # reduce_dimension = True
        # while reduce_dimension:
        #     predict = self._combine_components(predict)
        #     if predict[0].shape[0] == new_shape or predict[0].shape[0] == 1:
        #         reduce_dimension = False
        #     new_shape = predict[0].shape[0]
        # predict = self._clean_predict(np.array(v))
        return predict

    def _transform_one_sample(self, series: np.array, svd_flag: bool = False):
        trajectory_transformer = HankelMatrix(time_series=series, window_size=self.window_size)
        data = trajectory_transformer.trajectory_matrix
        self.window_size = trajectory_transformer.window_length
        self.decomposer = SpectrumDecomposer(data, trajectory_transformer.ts_length, self.SV_threshold)
        if svd_flag:
            return self.estimate_singular_values(data)
        return self._get_basis(data)

    def get_threshold(self, data, selector: str):

        selectors = {'median': stats.mode,
                     'mode': stats.mode}

        svd_numbers = []
        with tqdm(total=len(data), desc='SVD estimation') as pbar:
            for signal in data:
                svd_numbers.append(self._transform_one_sample(signal, svd_flag=True))
                pbar.update(1)

        return selectors[selector](svd_numbers).astype(int)

    def estimate_singular_values(self, data):
        basis = Either.insert(data).then(self.decomposer.svd).value[0]
        spectrum = [s_val for s_val in basis[1] if s_val > 0.001]
        # self.left_approx_sv, self.right_approx_sv = basis[0], basis[2]
        return len(spectrum)

    def _get_1d_basis(self, data):
        basis = Either.insert(data).then(self.decomposer.svd).then(self.decomposer.threshold).then(
            self.decomposer.data_driven_basis).value[0]
        return np.swapaxes(basis, 1, 0)

    def _get_multidim_basis(self, data):
        basis = np.array(
            Either.insert(data).then(self.decomposer.tensor_decomposition).then(self.decomposer.multi_threshold).then(
                self.decomposer.data_driven_basis).value[0])
        basis = basis.reshape(basis.shape[1], -1)
        return basis
