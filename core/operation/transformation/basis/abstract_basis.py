import numpy as np


class BasisDecomposition:
    """A class for decomposing data on the abstract basis and evaluating the derivative of the resulting decomposition.
    """

    def __init__(self, n_components: int = None):
        """
        Initializes the class with an array of data in np.array format as input.

        Parameters
        ----------
        """

        self.n_components = n_components
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

    def transform(self):
        pass
