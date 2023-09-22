from typing import Optional

import numpy as np
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.operation.transformation.basis.abstract_basis import BasisDecompositionImplementation


class FourierBasisImplementation(BasisDecompositionImplementation):
    """A class for decomposing data on the Fourier basis and evaluating the derivative of the resulting decomposition.

    Example::
        ts1 = np.random.rand(200)
        ts2 = np.random.rand(200)
        ts = [ts1, ts2]
        bss = FourierBasisImplementation({'threshold': 20000'})
        basis_multi = bss.transform(ts)
        basis_1d = bss.transform(ts1)

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.threshold = params.get('threshold')
        self.basis = None

        self.logging_params.update({'threshold': self.threshold})

    def _decompose_signal(self, input_data):
        fourier_coef = np.fft.rfft(input_data)
        frequencies = np.fft.rfftfreq(input_data.size, d=2e-3 / input_data.size)
        fourier_coef[frequencies > self.threshold] = 0
        return np.fft.irfft(fourier_coef)

    def _transform_one_sample(self, series: np.array):
        return self._get_basis(series)

    def evaluate_derivative(self, order):
        """Evaluates the derivative of the Fourier decomposition of the given data.

        Returns:
            np.array: The derivative of the Fourier decomposition of the given data.
        """
        return np.fft.ifft(1j * np.arange(len(self.data_range)) * self.decomposed)
