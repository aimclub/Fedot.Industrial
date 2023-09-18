from fedot_ind.core.operation.decomposition.matrix_decomposition.dmd_decomposition import *


class piDMD:
    """
    Computes a dynamic mode decomposition when the solution matrix is
    constrained to lie in a matrix manifold.
    The options available for the "method"
    - "exact", "exactSVDS"
    - "orthogonal"
    - "uppertriangular, "lowertriangular"
    - "diagonal", "diagonalpinv", "diagonaltls", "symtridiagonal"
    - "circulant", "circulantTLS", "circulantunitary", "circulantsymmetric",
    "circulantskewsymmetric"
    - "hankel", "toeplitz"
    - "symmetric", "skewsymmetric"
    """

    def __init__(self, method):
        self.method_dict = {"exact": exact_dmd_decompose,
                            "orthogonal": orthogonal_dmd_decompose,
                            'hankel': hankel_decompose,
                            'symmetric': symmetric_decompose
                            }
        self.method = method

    def fit(self, train_features, train_target):
        nx, nt = train_features.shape
        rank = min(nx, nt)
        self.fitted_linear_operator, eigenvals, eigenvectors = self.method_dict[self.method](train_features, train_target, rank)
        return self.fitted_linear_operator, eigenvals, eigenvectors

    def predict(self, test_features):
        return self.fitted_linear_operator(test_features)
        # elif method == 'uppertriangular':
        #     R, Q = rq(X)
        #     Ut = np.triu(Y @ Q.T)
        #     A = Ut / R
        # elif method == 'lowertriangular':
        #     A = np.rot90(piDMD(np.flipud(X), np.flipud(Y), 'uppertriangular'), 2)
        # elif method.startswith('diagonal'):
        #     if len(args) > 0:
        #         d = args[0]
        #         if len(d) == 1:
        #             d = d * np.ones((nx, 2))
        #         elif len(d) == nx:
        #             d = np.repeat(d[:, np.newaxis], 2, axis=1)
        #         elif any(d.shape != (nx, 2)):
        #             raise ValueError('Diagonal number is not in an allowable format.')
        #     else:
        #         d = np.ones((nx, 2))
        #     Icell = [None] * nx
        #     Jcell = [None] * nx
        #     Rcell = [None] * nx
        #     for j in range(nx):
        #         l1 = max(j - (d[j, 0] - 1), 0)
        #         l2 = min(j + (d[j, 1] - 1), nx - 1)
        #         C = X[l1:l2 + 1, :]
        #         b = Y[j, :]
        #         if method == 'diagonal':
        #             sol = b / C
        #         elif method == 'diagonalpinv':
        #             sol = b @ np.linalg.pinv(C)
        #         elif method == 'diagonaltls':
        #             sol = tls(C.T, b.T).T
        #         Icell[j] = j * np.ones(1 + l2 - l1)
        #         Jcell[j] = np.arange(l1, l2 + 1)
        #         Rcell[j] = sol
        #     Imat = np.concatenate(Icell)
        #     Jmat = np.concatenate(Jcell)
        #     Rmat = np.concatenate(Rcell)
        #     Asparse = scipy.sparse.csr_matrix((Rmat, (Imat, Jmat)), shape=(nx, nx))
        #     A = lambda v: Asparse @ v
        #     if nargout == 2:
        #         eVals = eigs(Asparse, nx)
        #         varargout[0] = eVals
        #     else:
        #         eVecs, eVals = eigs(Asparse, nx)
        #         varargout[0] = np.diag(eVals)
        #         varargout[1] = eVecs
        # elif method == 'symmetric' or method == 'skewsymmetric':
        #     Ux, S, V = svd(X, 0)
        #     C = Ux.T @ Y @ V
        #     C1 = C
        #     if len(args) > 0:
        #         r = args[0]
        #     else:
        #         r = np.linalg.matrix_rank(X)
