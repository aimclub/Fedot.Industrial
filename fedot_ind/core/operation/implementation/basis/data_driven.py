from typing import Tuple, TypeVar, Optional

import numpy as np
from fedot.core.operations.operation_parameters import OperationParameters
from pymonad.either import Either

from fedot_ind.core.operation.decomposition.SpectrumDecomposition import SpectrumDecomposer
from fedot_ind.core.operation.implementation.basis.abstract_basis import BasisDecompositionImplementation
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix

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

        self.decomposer = None
        self.basis = None

    def _transform_one_sample(self, series: np.array):
        trajectory_transformer = HankelMatrix(time_series=series, window_size=self.window_size)
        data = trajectory_transformer.trajectory_matrix
        self.ts_length = trajectory_transformer.ts_length
        self.decomposer = SpectrumDecomposer(data, self.n_components, self.ts_length)
        return self._get_basis(data)

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

