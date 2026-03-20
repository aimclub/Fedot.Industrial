from __future__ import annotations

import numpy as np

from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.okhs import (
    FractionalDMD as OKHSFractionalDMD,
    FractionalLiouvilleOperator,
    OKHSTransformer,
)


class _DefaultRBFKernel:
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def _compute_single_kernel(self, x, y):
        left = np.atleast_1d(np.asarray(x, dtype=float))
        right = np.atleast_1d(np.asarray(y, dtype=float))
        diff = left - right
        return float(np.exp(-self.gamma * np.dot(diff, diff)))


class FractionalDMD:
    """
    Legacy-compatible Fractional DMD wrapper aligned with the current OKHS core pipeline.
    """

    def __init__(
            self,
            q=0.7,
            n_modes=None,
            kernel=None,
            n_quad_points=20,
            dt=1.0,
            regularization=1e-8,
    ):
        self.q = q
        self.n_modes = n_modes
        self.kernel = kernel or _DefaultRBFKernel()
        self.n_quad_points = n_quad_points
        self.dt = dt
        self.regularization = regularization

        self.okhs = None
        self.liouville_operator_ = None
        self.fdmd_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.modes_ = None
        self.training_trajectories_ = None

    @staticmethod
    def _normalize_trajectory(trajectory):
        array = np.asarray(trajectory, dtype=float)
        if array.ndim == 0:
            return array.reshape(1, 1)
        if array.ndim == 1:
            return array.reshape(-1, 1)
        return array

    def _select_modes(self):
        if self.n_modes is None or self.eigenvalues_ is None:
            return

        self.eigenvalues_ = self.eigenvalues_[: self.n_modes]
        self.eigenvectors_ = self.eigenvectors_[:, : self.n_modes]
        self.modes_ = self.modes_[: self.n_modes]
        self.liouville_operator_.eigenvalues_ = self.eigenvalues_
        self.liouville_operator_.eigenvectors_ = self.eigenvectors_
        self.fdmd_.modes_ = self.modes_

    def fit(self, trajectories):
        normalized_trajectories = [self._normalize_trajectory(trajectory) for trajectory in trajectories]
        self.training_trajectories_ = normalized_trajectories

        self.okhs = OKHSTransformer(
            kernel=self.kernel,
            q=self.q,
            n_quad_points=self.n_quad_points,
            dt=self.dt,
        )
        self.okhs.fit(normalized_trajectories)

        self.liouville_operator_ = FractionalLiouvilleOperator(
            okhs_transformer=self.okhs,
            n_quad_points=self.n_quad_points,
        )
        self.liouville_operator_.fit()

        self.fdmd_ = OKHSFractionalDMD(
            liouville_operator=self.liouville_operator_,
            n_quad_points=self.n_quad_points,
            regularization=self.regularization,
        )
        self.fdmd_.fit(normalized_trajectories)

        self.eigenvalues_ = np.asarray(self.liouville_operator_.eigenvalues_)
        self.eigenvectors_ = np.asarray(self.liouville_operator_.eigenvectors_)
        self.modes_ = np.asarray(self.fdmd_.modes_)
        self._select_modes()
        return self

    def predict(self, initial_condition, time_steps):
        if self.fdmd_ is None:
            raise ValueError("FractionalDMD must be fitted before calling predict.")

        if np.isscalar(time_steps):
            times = np.arange(1, int(time_steps) + 1)
        else:
            times = np.asarray(time_steps, dtype=float)

        if np.isscalar(initial_condition):
            initial_trajectory = np.full((min(5, max(2, len(times))), 1), float(initial_condition))
        else:
            initial_trajectory = self._normalize_trajectory(initial_condition)

        prediction = self.fdmd_.predict(initial_trajectory, times)
        if prediction.ndim == 2 and prediction.shape[1] == 1:
            return prediction.flatten()
        return prediction
