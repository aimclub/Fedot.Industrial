import math
from typing import Optional, Union

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from joblib import cpu_count, delayed, Parallel
from pymonad.either import Either
from pymonad.list import ListMonad

from fedot_ind.core.operation.IndustrialCachableOperation import IndustrialCachableOperationImplementation


class BasisDecompositionImplementation(IndustrialCachableOperationImplementation):
    """
    A class for decomposing data on the abstract basis and evaluating the derivative of the resulting decomposition.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.n_processes = math.ceil(cpu_count() * 0.7) if cpu_count() > 1 else 1
        self.n_components = params.get('n_components', 2)
        self.basis = None
        self.data_type = DataTypesEnum.image
        self.min_rank = 1

        self.logging_params = {'jobs': self.n_processes}

    def _get_basis(self, data):
        basis = Either.insert(data).then(self._get_1d_basis if type(data) != list else self._get_multidim_basis).value
        return basis

    def fit(self, data):
        """Decomposes the given data on the chosen basis.

        Returns:
            np.array: The decomposition of the given data.
        """
        pass

    def _decompose_signal(self, signal) -> list:
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

    def _transform(self, input_data: Union[InputData, pd.DataFrame]) -> np.array:
        """Method for transforming all samples

        """
        if type(input_data) is InputData:
            features = np.array(ListMonad(*input_data.features.tolist()).value)
        else:
            features = np.array(ListMonad(*input_data.values.tolist()).value)
        features = np.array([series[~np.isnan(series)] for series in features])

        parallel = Parallel(n_jobs=self.n_processes, verbose=0, pre_dispatch="2*n_jobs")
        v = parallel(delayed(self._transform_one_sample)(sample) for sample in features)

        predict = np.array(v)
        return predict

    def _get_multidim_basis(self, input_data):
        decompose = lambda multidim_signal: ListMonad(list(map(self._decompose_signal, multidim_signal)))
        basis = Either.insert(input_data).then(decompose).value[0]
        return basis
