import numpy as np
from typing import Tuple, TypeVar
from fedot_ind.core.operation.transformation.basis.abstract_basis import BasisDecompositionImplementation

class_type = TypeVar("T", bound="PowerBasis")


class PowerBasis(BasisDecompositionImplementation):
    """Power basis.
    Basis formed by powers of the argument :math:`t`:
    .. math::
        1, t, t^2, t^3...
    Attributes:
        data_range: a tuple of length 2 containing the initial and
            end values of the interval over which the basis can be evaluated.
        n_components: number of functions in the basis.
    """

    def _get_basis(self, n_components: int = None):
        self.basis = np.arange(self.n_components)

        return self.basis

    def fit(self, data):
        projected_data = np.power.outer(data, self.basis)
        return projected_data

    def evaluate_derivative(self: class_type,
                            coefs: np.array,
                            order: int = 1) -> Tuple[class_type, np.array]:
        basis = type(self)(
            domain_range=self.domain_range,
            n_basis=self.n_basis - order,
        )
        derivative_coefs = np.array([np.polyder(x[::-1], order)[::-1] for x in coefs])

        return basis, derivative_coefs


