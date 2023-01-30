import numpy as np
from core.operation.transformation.basis.abstract_basis import BasisDecomposition


class FourierBasis(BasisDecomposition):
    """A class for decomposing data on the Fourier basis and evaluating the derivative of the resulting decomposition.

    Attributes:
        data (np.array): The array of data to be decomposed.
    """

    def _get_basis(self, n_components: int = None):
        return np.fft.fft

    def fit(self, data):
        """Decomposes the given data on the Fourier basis.

        Returns:
            np.array: The Fourier decomposition of the given data.
        """
        self.decomposed = np.fft.fft(data)
        return self.basis(data)

    def evaluate_derivative(self, order):
        """Evaluates the derivative of the Fourier decomposition of the given data.

        Returns:
            np.array: The derivative of the Fourier decomposition of the given data.
        """
        return np.fft.ifft(1j * np.arange(len(self.data_range)) * self.decomposed)
