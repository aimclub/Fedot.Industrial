import numpy as np
from pymonad.list import ListMonad

from core.operation.transformation.basis.abstract_basis import BasisDecomposition


class FourierBasis(BasisDecomposition):
    """A class for decomposing data on the Fourier basis and evaluating the derivative of the resulting decomposition.

    Attributes:
        data (np.array): The array of data to be decomposed.
    """

    def _get_basis(self, n_components: int = None):
        return np.fft.fft

    def _transform(self, features: ListMonad):
        self.decomposed = np.fft.fft(features)
        return self.basis(features)

    def evaluate_derivative(self, order):
        """Evaluates the derivative of the Fourier decomposition of the given data.

        Returns:
            np.array: The derivative of the Fourier decomposition of the given data.
        """
        return np.fft.ifft(1j * np.arange(len(self.data_range)) * self.decomposed)
