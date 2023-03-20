import os
from copy import copy
from typing import Optional

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation, _convert_to_output_function
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from pymonad.list import ListMonad

from core.architecture.utils.utils import PROJECT_PATH
from core.operation.utils.cache import DataCacher


class BasisDecompositionImplementation(DataOperationImplementation):
    """A class for decomposing data on the abstract basis and evaluating the derivative of the resulting decomposition.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        """
        Initializes the class with an array of data in np.array format as input.

        Parameters
        ----------
        """

        super().__init__(params)
        self.n_components = params.get('n_components', 2)
        self.basis = None

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

    def evaluate_derivative(self, order: int = 1):
        """Evaluates the derivative of the decomposition of the given data.

        Returns:
            np.array: The derivative of the decomposition of the given data.
        """
        pass

    def f(self, x):
        return self._transform(x)

    def transform(self, input_data: InputData) -> OutputData:
        features = ListMonad(*input_data.features.tolist()).value

        cache_folder = os.path.join(PROJECT_PATH, 'cache')
        os.makedirs(cache_folder, exist_ok=True)
        cacher = DataCacher(data_type_prefix=f'Features of  basis',
                            cache_folder=cache_folder)

        hashed_info = cacher.hash_info(data=input_data.features.tobytes(),
                                       **self.params.to_dict())

        try:
            predict = cacher.load_data_from_cache(hashed_info=hashed_info)
        except FileNotFoundError:
            v = np.vectorize(self.f, signature='(n)->(m, n)')

            predict = v(features)
        predict = self._convert_to_output(input_data, predict, data_type=DataTypesEnum.image)
        return predict

    def _transform(self, series: np.array):
        pass

    def _get_multidim_basis(self):
        pass

    def _get_1d_basis(self):
        pass

