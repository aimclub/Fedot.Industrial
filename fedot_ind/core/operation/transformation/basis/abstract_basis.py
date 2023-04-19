from typing import Optional
import numpy as np
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from pymonad.either import Either
from pymonad.list import ListMonad
from tqdm.auto import tqdm

from fedot_ind.core.operation.IndustrialCachableOperation import IndustrialCachableOperationImplementation


class BasisDecompositionImplementation(IndustrialCachableOperationImplementation):
    """
    A class for decomposing data on the abstract basis and evaluating the derivative of the resulting decomposition.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.n_components = params.get('n_components', 2)
        self.basis = None
        self.data_type = DataTypesEnum.image
        self.min_rank = 1

    def _get_basis(self, data):
        if type(data) == list:
            basis = self._get_multidim_basis(data)
        else:
            basis = self._get_1d_basis(data)
        return basis

    def fit(self, data):
        """Decomposes the given data on the chosen basis.

        Returns:
            np.array: The decomposition of the given data.
        """
        pass

    def _decompose_signal(self):
        pass

    def evaluate_derivative(self, order: int = 1):
        """Evaluates the derivative of the decomposition of the given data.

        Returns:
            np.array: The derivative of the decomposition of the given data.
        """
        pass

    def _transform_one_sample(self, sample: np.array):
        """
            Method for transforming one sample
        """
        pass

    def _get_1d_basis(self, input_data):
        decompose = lambda signal: ListMonad(self._decompose_signal(signal))
        basis = Either.insert(input_data).then(decompose).value[0]
        return basis

    def _transform(self, input_data: InputData) -> np.array:
        """
            Method for transforming all samples
        """
        features = np.array(ListMonad(*input_data.features.tolist()).value)
        v = []
        for series in tqdm(features):
            v.append(self._transform_one_sample(series[~np.isnan(series)]))
        predict = np.array(v)
        return predict

    def _get_multidim_basis(self, data):
        pass

    def _get_multidim_basis(self, input_data):
        decompose = lambda multidim_signal: ListMonad(list(map(self._decompose_signal, multidim_signal)))
        basis = Either.insert(input_data).then(decompose).value[0]
        return basis
