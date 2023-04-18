import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from fedot_ind.core.metrics.loss.basis_loss import basis_approximation_metric
from fedot_ind.core.operation.transformation.basis.abstract_basis import BasisDecompositionImplementation


class LegenderBasis(BasisDecompositionImplementation):

    def _get_basis(self, data, n_components: int = None):
        self.basis = []
        self.coefs = []
        self.rmse = []
        domain = np.arange(self.data_range)
        for i in range(1, self.n_components + 1):
            legendre_coefs = np.polynomial.legendre.legfit(x=domain, y=data, deg=i)
            new_y = np.polynomial.legendre.legval(domain, legendre_coefs)
            self.coefs.append(legendre_coefs)
            self.basis.append(new_y)
            self.rmse.append(mean_squared_error(y_true=data,
                                                y_pred=new_y,
                                                squared=False))

    def fit(self, data):
        # Create the orthogonal polynomials
        self._get_basis(data, self.n_components)
        self.data = data

    def evaluate_derivative(self, order):
        domain = np.arange(self.data_range)
        self.derv_coef = []
        for poly_coef in self.coefs:
            derivative_of_poly = np.polynomial.legendre.legder(poly_coef, order)
            new_y = np.polynomial.legendre.legval(domain, derivative_of_poly)
            self.derv_coef.append(new_y)

    def analytical_form(self):
        polynom_number = basis_approximation_metric(metric_values=np.array(self.rmse),
                                                    derivation_coef=self.derv_coef,
                                                    regularization_coef=0.1)
        polynom_coef = self.coefs[polynom_number]
        p = np.polynomial.legendre.leg2poly(polynom_coef)
        anylytical_form = ''
        for index, coef in enumerate(p):
            degree = f' +{round(coef, 3)}*x^({len(p) - index - 1}) '
            anylytical_form = anylytical_form + degree
        anylytical_form = anylytical_form.replace('+-', '-')
        print(f'Best degree of Legendre Polynomial is - {polynom_number + 1}')
        print(anylytical_form)
        return polynom_number

    def show(self, visualisation_type: str = 'basis representation', **kwargs):
        if visualisation_type == 'basis representation':
            A = np.array(self.basis).T
        else:
            A = np.array(self.derv_coef).T
        A = pd.DataFrame(A)
        A['original'] = self.data
        A[[kwargs['basis_function'], 'original']].plot()
        plt.show()

    # def transform(self, X):
    #     # Create the orthogonal polynomials
    #
    #
    #     # Create the matrix of polynomials
    #     A = np.array(polys).T
    #
    #     # Transform the data using the coefficients
    #     return A.dot(self.coefficients)
