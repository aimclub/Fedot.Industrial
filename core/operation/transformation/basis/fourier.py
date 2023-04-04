from typing import Optional
import numpy as np
from fedot.core.operations.operation_parameters import OperationParameters
from core.operation.transformation.basis.abstract_basis import BasisDecompositionImplementation


class FourierBasisImplementation(BasisDecompositionImplementation):
    """A class for decomposing data on the Fourier basis and evaluating the derivative of the resulting decomposition.
        Example:
            ts1 = np.random.rand(200)
            ts2 = np.random.rand(200)
            ts = [ts1, ts2]
            bss = FourierBasisImplementation({'spectrum_type': 'real'})
            basis_multi = bss._transform(ts)
            basis_1d = bss._transform(ts1)

    Attributes:
        data (np.array): The array of data to be decomposed.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.spectrum_type = params.get('spectrum_type')
        self.basis = None

    def _decompose_signal(self, input_data):
        spectrum = np.fft.fft(input_data)
        if self.spectrum_type == 'imaginary':
            spectrum = spectrum.imag
        else:
            spectrum = spectrum.real
        # freq = np.fft.fftfreq(input_data.shape[-1])
        return spectrum

    def _transform(self, series: np.array):
        return self._get_basis(series)

    def evaluate_derivative(self, order):
        """Evaluates the derivative of the Fourier decomposition of the given data.

        Returns:
            np.array: The derivative of the Fourier decomposition of the given data.
        """
        return np.fft.ifft(1j * np.arange(len(self.data_range)) * self.decomposed)