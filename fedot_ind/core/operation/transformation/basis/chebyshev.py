import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from fedot_ind.core.metrics.loss.basis_loss import basis_approximation_metric
from fedot_ind.core.operation.transformation.basis.abstract_basis import BasisDecompositionImplementation


class ChebyshevBasis(BasisDecompositionImplementation):

    def _get_basis(self, data, n_components: int = None):
        self.basis = []
        self.coefs = []
        self.rmse = []
        time_domain = np.arange(self.data_range)
        value_domain = np.arange(np.min(data), np.min(data), 0.1, dtype=float)
        for i in range(1, self.n_components + 1):
            # poly_coefs = np.polynomial.chebyshev.chebgrid2d(x=time_domain, y=value_domain, c=i)
            poly_coefs = np.polynomial.chebyshev.chebfit(x=time_domain, y=data, deg=i)
            # poly_coefs = np.ones(shape=len(time_domain))
            new_y = np.polynomial.chebyshev.chebval(time_domain, poly_coefs)
            self.coefs.append(poly_coefs)
            self.basis.append(new_y)
            self.rmse.append(mean_squared_error(y_true=data,
                                                y_pred=new_y,
                                                squared=False))

    def _transform(self, features):
        self._get_basis(features, self.n_components)
        self.basis = np.array(self.basis).T
        self.data = features
        # coef =Ridge(alpha=0.5).fit(self.basis, self.data.T)
        return self.basis

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
                                                    regularization_coef=0.9)
        polynom_coef = self.coefs[polynom_number]
        p = np.polynomial.chebyshev.cheb2poly(polynom_coef)
        anylytical_form = ''
        for index, coef in enumerate(p):
            degree = f' +{round(coef, 3)}*x^({len(p) - index - 1}) '
            anylytical_form = anylytical_form + degree
        anylytical_form = anylytical_form.replace('+-', '-')
        print(f'Best degree of Chebyshev Polynomial is - {polynom_number + 1}')
        print(anylytical_form)
        return polynom_number

    def show(self, visualisation_type: str = 'basis representation', **kwargs):
        if visualisation_type == 'basis representation':
            A = np.array(self.basis)
        else:
            A = np.array(self.derv_coef)
        A = pd.DataFrame(A)
        A['original'] = self.data
        A[[kwargs['basis_function'], 'original']].plot()
        plt.show()
