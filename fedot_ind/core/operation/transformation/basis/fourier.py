from typing import Optional
import numpy as np
from fedot.core.operations.operation_parameters import OperationParameters
from pymonad.either import Either
from pymonad.list import ListMonad

from fedot_ind.core.operation.transformation.basis.abstract_basis import BasisDecompositionImplementation


class FourierBasisImplementation(BasisDecompositionImplementation):
    """A class for decomposing data on the Fourier basis and evaluating the derivative of the resulting decomposition.

    Example::
        ts1 = np.random.rand(200)
        ts2 = np.random.rand(200)
        ts = [ts1, ts2]
        bss = FourierBasisImplementation({'spectrum_type': 'real'})
        basis_multi = bss._transform(ts)
        basis_1d = bss._transform(ts1)

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.spectrum_type = params.get('spectrum_type')
        self.threshold = params.get('threshold')
        self.basis = None

        self.logging_params.update({'spectr': self.spectrum_type, 'thresh': self.threshold})

    def _get_1d_basis(self, input_data):
        decompose = lambda signal: ListMonad(self._decompose_signal(signal))
        basis = Either.insert(input_data).then(decompose).value[0]
        return basis.reshape(1, -1)

    def low_pass(self, input_data):
        fourier_coef = np.fft.rfft(input_data)
        frequencies = np.fft.rfftfreq(input_data.size, d=2e-3 / input_data.size)
        fourier_coef[frequencies > self.threshold] = 0
        return np.fft.irfft(fourier_coef)

    def _decompose_signal(self, input_data):
        spectrum = np.fft.fft(input_data)
        if self.spectrum_type == 'imaginary':
            spectrum = spectrum.imag
        elif self.spectrum_type == 'smoothed':
            spectrum = self.low_pass(input_data)
        elif self.spectrum_type == 'real':
            spectrum = spectrum.real
        else:
            raise ValueError('Unknown spectrum type. Must be either ``real``, ``imaginary`` or ``smoothed``')
        return spectrum

    def _transform_one_sample(self, series: np.array):
        return self._get_basis(series)

    def evaluate_derivative(self, order):
        """Evaluates the derivative of the Fourier decomposition of the given data.

        Returns:
            np.array: The derivative of the Fourier decomposition of the given data.
        """
        return np.fft.ifft(1j * np.arange(len(self.data_range)) * self.decomposed)
