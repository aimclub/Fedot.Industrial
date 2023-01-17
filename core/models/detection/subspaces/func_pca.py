from typing import Optional, Any, Union

import numpy as np
import scipy.integrate
from scipy.linalg import solve_triangular
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from sktime.distances import pairwise_distance

from core.architecture.preprocessing.DatasetLoader import DataLoader
from core.operation.transformation.basis.abstract_basis import BasisDecomposition
from core.operation.transformation.basis.chebyshev import ChebyshevBasis


# from core.operation.transformation.regularization.lp_reg import compute_penalty_matrix


class FunctionalPCA:
    """
    Principal component analysis.
    Class that implements functional principal component analysis for both
    basis and natural representations of the data.

    Parameters:
        n_components: Number of principal components to keep from
            functional principal component analysis. Defaults to 3.
        regularization: Regularization object to be applied.
        basis_of_function: .
    Attributes:
        components\_: this contains the principal components.
        explained_variance\_ : The amount of variance explained by
            each of the selected components.
        explained_variance_ratio\_ : this contains the percentage
            of variance explained by each principal component.
        singular_values\_: The singular values corresponding to each of the
            selected components.
        mean\_: mean of the train data.
    Examples:

    time_series = np.array([1,2,3,4,5,6])
    data_range = len(time_series)
    basis = ChebyshevBasis(data_range=data_range, n_components=4).decompose(time_series)
    FPCA = FunctionalPCA(2)
    FPCA = FPCA.fit(basis)

    """

    def __init__(
            self,
            n_components: int = 3,
            regularization: callable = None,
            basis_function: Union[np.array, BasisDecomposition] = None,
            # _weights: Optional[Union[ArrayLike, WeightsCallable]] = None,
    ) -> None:
        self.n_components = n_components
        self.regularization = regularization
        # self._weights = _weights
        self.basis_function = basis_function

    def _fit_basis(
            self,
            X: np.ndarray,
            y: object = None,
    ):
        """
        Compute the first n_components principal components and saves them.
        Args:
            X: The functional data object to be analysed.
            y: Ignored.
        Returns:
            self
        References:
            .. [RS05-8-4-2] Ramsay, J., Silverman, B. W. (2005). Basis function
                expansion of the functions. In *Functional Data Analysis*
                (pp. 161-164). Springer.
        """

        mean_centred = np.allclose(X, X - np.mean(X, axis=0))
        if not mean_centred:
            X = X - np.mean(X, axis=0)
        gram = pairwise_distance(X.T)
        if self.basis_function is not None:
            # Compute Gram matrix of basis function
            G = self.basis_function.T.dot(self.basis_function)
            # The matrix that are in charge of changing the computed principal
            # components to target matrix is essentially the inner product
            # of both basis.
            J = np.dot(X, self.basis_function)
        else:
            # If no other basis is specified we use the same basis as the
            # passed FDataBasis object
            # components_basis = X.copy()
            G = X.T.dot(X)
            J = G

        self._X_basis = X
        self._j_matrix = J

        # # Apply regularization
        # if self.regularization is not None:
        #     regularization_matrix = compute_penalty_matrix(
        #         basis_iterable=(components_basis,),
        #         regularization_parameter=1,
        #         regularization=self.regularization,
        #     )
        #
        #     G = G + regularization_matrix

        # Diagonalisation of Gram Matrix. G = L*L^T
        l_matrix = np.linalg.cholesky(G)

        # we need L^{-1} for a multiplication, there are two possible ways:
        # using solve to get the multiplication result directly or just invert
        # the matrix. We choose solve because it is faster and more stable.
        # The following matrix is needed: L^{-1}*J^T
        l_inv_j_t = solve_triangular(
            l_matrix,
            np.transpose(J),
            lower=True,
        )

        # the final matrix, C(L-1Jt)t for svd or (L-1Jt)-1CtC(L-1Jt)t for PCA

        final_matrix = X @ np.transpose(l_inv_j_t)

        # initialize the pca module provided by scikit-learn
        pca = PCA(n_components=self.n_components)
        pca.fit(final_matrix)

        # we choose solve to obtain the component coefficients for the
        # same reason: it is faster and more efficient
        component_coefficients = solve_triangular(
            np.transpose(l_matrix),
            np.transpose(pca.components_),
            lower=False,
        )

        self.explained_variance_ratio = pca.explained_variance_ratio_
        self.explained_variance_ = pca.explained_variance_
        self.singular_values_ = pca.singular_values_
        # self.components_ = X.copy(
        #     basis=components_basis,
        #     coefficients=component_coefficients.T,
        #     sample_names=(None,) * self.n_components,
        # )

        return self

    def _transform_basis(
            self,
            X: np.array
    ):
        """Compute the n_components first principal components score.
        Args:
            X: The functional data object to be analysed.
        Returns:
            Principal component scores.
        """

        # Compute inner product of our data with the components
        return X.coefficients @ self._j_matrix @ self.components_.coefficients.T

    def _fit_grid(
            self,
            X: np.array,
            y: object = None,
    ):
        r"""
        Compute the n_components first principal components and saves them.
        The eigenvalues associated with these principal
        components are also saved. For more details about how it is implemented
        please view the referenced book, chapter 8.
        In summary, we are performing standard multivariate PCA over
        :math:`\mathbf{X} \mathbf{W}^{1/2}` where :math:`\mathbf{X}` is the
        data matrix and :math:`\mathbf{W}` is the weight matrix (this matrix
        defines the numerical integration). By default the weight matrix is
        obtained using the trapezoidal rule.
        Args:
            X: The functional data object to be analysed.
            y: Ignored.
        Returns:
            self.
        References:
            .. [RS05-8-4-1] Ramsay, J., Silverman, B. W. (2005). Discretizing
                the functions. In *Functional Data Analysis* (p. 161).
                Springer.
        """

        # data matrix initialization
        fd_data = X.data_matrix.reshape(X.data_matrix.shape[:-1])

        # get the number of samples and the number of points of descretization
        n_samples, n_points_discretization = fd_data.shape

        # necessary for inverse_transform
        self.n_samples_ = n_samples

        # if centering is True then subtract the mean function to each function
        # in FDataBasis
        X = self._center_if_necessary(X)

        # establish weights for each point of discretization
        if self._weights is None:
            # grid_points is a list with one array in the 1D case
            identity = np.eye(len(X.grid_points[0]))
            self._weights = scipy.integrate.simps(identity, X.grid_points[0])
        elif callable(self._weights):
            self._weights = self._weights(X.grid_points[0])
            # if its a FDataGrid then we need to reduce the dimension to 1-D
            # array
            if isinstance(self._weights, np.array):
                self._weights = np.squeeze(self._weights.data_matrix)
        else:
            self._weights = self._weights

        weights_matrix = np.diag(self._weights)

        # basis = FDataGrid(
        #     data_matrix=np.identity(n_points_discretization),
        #     grid_points=X.grid_points,
        # )

        regularization_matrix = compute_penalty_matrix(
            basis_iterable=(np.array,),
            regularization_parameter=1,
            regularization=self.regularization,
        )

        # See issue #497 for more information about this approach
        factorization_matrix = weights_matrix.astype(float)
        if self.regularization is not None:
            factorization_matrix += regularization_matrix

        # Tranpose of the Cholesky decomposition
        Lt = np.linalg.cholesky(factorization_matrix).T

        new_data_matrix = fd_data @ weights_matrix
        new_data_matrix = np.linalg.solve(Lt.T, new_data_matrix.T).T

        pca = PCA(n_components=self.n_components)
        pca.fit(new_data_matrix)

        components = pca.components_
        components = np.linalg.solve(Lt, pca.components_.T).T

        self.components_ = X.copy(
            data_matrix=components,
            sample_names=(None,) * self.n_components,
        )

        self.explained_variance_ratio_ = (
            pca.explained_variance_ratio_
        )
        self.explained_variance_ = pca.explained_variance_
        self.singular_values_ = pca.singular_values_

        return self

    def _transform_grid(
            self,
            X: np.array,
            y: object = None,
    ):
        """
        Compute the ``n_components`` first principal components score.
        Args:
            X: The functional data object to be analysed.
            y: Ignored.
        Returns:
            Principal component scores.
        """
        # in this case its the coefficient matrix multiplied by the principal
        # components as column vectors

        return (  # type: ignore[no-any-return]
                X.data_matrix.reshape(X.data_matrix.shape[:-1])
                * self._weights
                @ np.transpose(
            self.components_.data_matrix.reshape(
                self.components_.data_matrix.shape[:-1],
            ),
        )
        )

    def fit(
            self,
            X: np.array
    ):
        """
        Compute the n_components first principal components and saves them.
        Args:
            X: The functional data object to be analysed.
        Returns:
            self
        """

        return self._fit_basis(X)

    def transform(
            self,
            X: np.array
    ):
        """
        Compute the ``n_components`` first principal components scores.
        Args:
            X: The functional data object to be analysed.
        Returns:
            Principal component scores.
        """

        return self._transform_basis(X)

    def fit_transform(
            self,
            X: np.array,
    ):
        """
        Compute the n_components first principal components and their scores.
        Args:
            X: The functional data object to be analysed.
        Returns:
            Principal component scores.
        """
        return self.fit(X).transform(X)

    def inverse_transform(
            self,
            pc_scores,
    ):
        """
        Compute the recovery from the fitted principal components scores.
        In other words,
        it maps ``pc_scores``, from the fitted functional PCs' space,
        back to the input functional space.
        ``pc_scores`` might be an array returned by ``transform`` method.
        Args:
            pc_scores: ndarray (n_samples, n_components).
        Returns:
            A FData object.
        """
        # check the instance is fitted.

        # input format check:
        if isinstance(pc_scores, np.ndarray):
            if pc_scores.ndim == 1:
                pc_scores = pc_scores[np.newaxis, :]

            if pc_scores.shape[1] != self.n_components:
                raise AttributeError(
                    "pc_scores must be a numpy array "
                    "with n_samples rows and n_components columns.",
                )
        else:
            raise AttributeError("pc_scores is not a numpy array.")

        # inverse_transform is slightly different whether
        # .fit was applied to FDataGrid or FDataBasis object
        # Does not work (boundary problem in x_hat and bias reconstruction)
        if isinstance(self.components_, np.array):

            additional_args = {
                "data_matrix": np.einsum(
                    "nc,c...->n...",
                    pc_scores,
                    self.components_.data_matrix,
                ),
            }

        elif isinstance(self.components_, np.array):

            additional_args = {
                "coefficients": pc_scores @ self.components_.coefficients,
            }

        return (
                self.mean_.copy(
                    **additional_args,
                    sample_names=(None,) * len(pc_scores),
                )
                + self.mean_
        )


dataset_name = 'Chinatown'
train, test = DataLoader(dataset_name).load_data()
train_target = train[1]
test_target = test[1]
time_series = train[0].iloc[1, :].values.flatten()
data_range = len(time_series)
basis = ChebyshevBasis(data_range=data_range, n_components=4).decompose(time_series)
FPCA = FunctionalPCA(n_components=2)
FPCA = FPCA.fit(basis)
