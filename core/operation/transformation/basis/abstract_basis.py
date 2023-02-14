from typing import Optional

import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from pymonad.list import ListMonad


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

    def _get_basis(self, **kwargs):
        """Defines the type of basis and the number of basis functions involved in the decomposition.

        """
        pass

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

    def transform(self, input_data: InputData):
        features = ListMonad(*input_data.features.tolist()).value
        output = np.array(self._transform(features))
        output = np.swapaxes(output, 1, 2)
        return output

    def _transform(self, features: ListMonad):
        pass
