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
            mode_selection_policy="fixed",
            mode_energy_threshold=0.95,
            prediction_mode_selection_policy="adaptive_tail_energy",
            max_prediction_modes=None,
            min_prediction_modes=4,
            boundary_alignment_policy="tapered_offset",
            boundary_alignment_decay=4.0,
            prediction_stability_threshold=0.03,
            kernel=None,
            n_quad_points=20,
            dt=1.0,
            regularization=1e-8,
    ):
        self.q = q
        self.n_modes = n_modes
        self.mode_selection_policy = mode_selection_policy
        self.mode_energy_threshold = mode_energy_threshold
        self.prediction_mode_selection_policy = prediction_mode_selection_policy
        self.max_prediction_modes = max_prediction_modes
        self.min_prediction_modes = min_prediction_modes
        self.boundary_alignment_policy = boundary_alignment_policy
        self.boundary_alignment_decay = boundary_alignment_decay
        self.prediction_stability_threshold = prediction_stability_threshold
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
        self.resolved_n_modes_ = None
        self.last_prediction_diagnostics_ = None
        self.last_fit_diagnostics_ = None

    @staticmethod
    def _normalize_trajectory(trajectory):
        array = np.asarray(trajectory, dtype=float)
        if array.ndim == 0:
            return array.reshape(1, 1)
        if array.ndim == 1:
            return array.reshape(-1, 1)
        return array

    def _resolve_selected_mode_count(self):
        if self.eigenvalues_ is None:
            return None

        total_modes = len(self.eigenvalues_)
        if total_modes == 0:
            return 0

        if str(self.mode_selection_policy).lower() == "energy":
            spectral_energy = np.abs(np.asarray(self.eigenvalues_, dtype=np.complex128))
            total_energy = float(np.sum(spectral_energy))
            if total_energy <= 0:
                return total_modes
            cumulative = np.cumsum(spectral_energy) / total_energy
            return int(max(1, np.searchsorted(cumulative, self.mode_energy_threshold, side="left") + 1))

        if self.n_modes is None:
            return total_modes
        return int(max(1, min(int(self.n_modes), total_modes)))

    def _select_modes(self):
        selected_mode_count = self._resolve_selected_mode_count()
        self.resolved_n_modes_ = selected_mode_count
        if selected_mode_count is None or self.eigenvalues_ is None:
            return

        self.eigenvalues_ = self.eigenvalues_[: selected_mode_count]
        self.eigenvectors_ = self.eigenvectors_[:, : selected_mode_count]
        self.modes_ = self.modes_[: selected_mode_count]
        self.liouville_operator_.eigenvalues_ = self.eigenvalues_
        self.liouville_operator_.eigenvectors_ = self.eigenvectors_
        self.fdmd_.modes_ = self.modes_

    @staticmethod
    def _flatten_feature_vector(values):
        array = np.asarray(values, dtype=float)
        return array.reshape(-1)

    def _apply_boundary_alignment(self, prediction, diagnostics, initial_trajectory):
        normalized_prediction = np.asarray(prediction, dtype=float)
        if normalized_prediction.ndim == 1:
            normalized_prediction = normalized_prediction.reshape(-1, 1)

        normalized_initial = self._normalize_trajectory(initial_trajectory)
        last_observed = self._flatten_feature_vector(normalized_initial[-1])
        raw_first_prediction = (
            self._flatten_feature_vector(normalized_prediction[0])
            if len(normalized_prediction)
            else last_observed.copy()
        )
        offset = last_observed - raw_first_prediction

        diagnostics = dict(diagnostics)
        diagnostics["boundary_alignment_policy"] = self.boundary_alignment_policy
        diagnostics["raw_first_prediction_value"] = raw_first_prediction.tolist()
        diagnostics["boundary_alignment_offset"] = offset.tolist()

        if self.boundary_alignment_policy in {"last_observation_offset", "tapered_offset"} and len(
                normalized_prediction):
            if self.boundary_alignment_policy == "last_observation_offset":
                alignment_weights = np.ones(len(normalized_prediction), dtype=float)
            else:
                alignment_weights = np.exp(
                    -np.linspace(0.0, float(self.boundary_alignment_decay), len(normalized_prediction), dtype=float)
                )
            aligned_prediction = normalized_prediction + alignment_weights[:, None] * offset.reshape(1, -1)
            corrected_first_prediction = self._flatten_feature_vector(aligned_prediction[0])
            diagnostics["boundary_alignment_applied"] = True
            diagnostics["boundary_alignment_weights"] = alignment_weights.tolist()
            diagnostics["corrected_first_prediction_value"] = corrected_first_prediction.tolist()
            diagnostics["boundary_discontinuity_after"] = (corrected_first_prediction - last_observed).tolist()
            diagnostics["boundary_discontinuity_after_abs_mean"] = float(
                np.mean(np.abs(corrected_first_prediction - last_observed))
            )
            return aligned_prediction, diagnostics

        diagnostics["boundary_alignment_applied"] = False
        diagnostics["corrected_first_prediction_value"] = raw_first_prediction.tolist()
        diagnostics["boundary_discontinuity_after"] = (raw_first_prediction - last_observed).tolist()
        diagnostics["boundary_discontinuity_after_abs_mean"] = float(
            np.mean(np.abs(raw_first_prediction - last_observed))
        )
        return normalized_prediction, diagnostics

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
        self.last_fit_diagnostics_ = self.get_fit_diagnostics_summary()
        return self

    def predict(self, initial_condition, time_steps):
        prediction, diagnostics = self.predict_with_diagnostics(initial_condition, time_steps)
        self.last_prediction_diagnostics_ = diagnostics
        return prediction

    def predict_with_diagnostics(self, initial_condition, time_steps):
        if self.fdmd_ is None:
            raise ValueError("FractionalDMD must be fitted before calling predict.")

        if np.isscalar(time_steps):
            horizon = int(time_steps)
            if np.isscalar(initial_condition):
                times = np.arange(1, horizon + 1, dtype=float) * self.dt
            else:
                normalized = self._normalize_trajectory(initial_condition)
                start_time = len(normalized) * self.dt
                stop_time = start_time + horizon * self.dt
                times = np.arange(start_time, stop_time, self.dt)
        else:
            times = np.asarray(time_steps, dtype=float)

        if np.isscalar(initial_condition):
            initial_trajectory = np.full((min(5, max(2, len(times))), 1), float(initial_condition))
        else:
            initial_trajectory = self._normalize_trajectory(initial_condition)

        if hasattr(self.fdmd_, "predict_with_diagnostics"):
            try:
                prediction, diagnostics = self.fdmd_.predict_with_diagnostics(
                    initial_trajectory,
                    times,
                    stability_threshold=self.prediction_stability_threshold,
                    prediction_mode_selection_policy=self.prediction_mode_selection_policy,
                    max_prediction_modes=self.max_prediction_modes,
                    min_prediction_modes=self.min_prediction_modes,
                )
            except TypeError:
                prediction, diagnostics = self.fdmd_.predict_with_diagnostics(
                    initial_trajectory,
                    times,
                )
        else:
            try:
                prediction = self.fdmd_.predict(
                    initial_trajectory,
                    times,
                    stability_threshold=self.prediction_stability_threshold,
                )
            except TypeError:
                prediction = self.fdmd_.predict(
                    initial_trajectory,
                    times,
                )
            diagnostics = {}
        aligned_prediction, diagnostics = self._apply_boundary_alignment(
            prediction=prediction,
            diagnostics=diagnostics,
            initial_trajectory=initial_trajectory,
        )
        if aligned_prediction.ndim == 2 and aligned_prediction.shape[1] == 1:
            aligned_prediction = aligned_prediction.flatten()

        diagnostics = {
            **self.get_fit_diagnostics_summary(),
            "prediction_time_grid": np.asarray(times, dtype=float).tolist(),
            **diagnostics,
        }
        self.last_prediction_diagnostics_ = diagnostics
        return aligned_prediction, diagnostics

    def get_fit_diagnostics_summary(self):
        eigenvalues = np.asarray(self.eigenvalues_) if self.eigenvalues_ is not None else np.array([])
        mode_norms = np.linalg.norm(self.modes_, axis=1).astype(float).tolist() if self.modes_ is not None else []
        return {
            "mode_selection_policy": self.mode_selection_policy,
            "mode_energy_threshold": self.mode_energy_threshold,
            "prediction_mode_selection_policy": self.prediction_mode_selection_policy,
            "max_prediction_modes": self.max_prediction_modes,
            "min_prediction_modes": self.min_prediction_modes,
            "boundary_alignment_policy": self.boundary_alignment_policy,
            "boundary_alignment_decay": self.boundary_alignment_decay,
            "prediction_stability_threshold": self.prediction_stability_threshold,
            "requested_n_modes": self.n_modes,
            "resolved_n_modes": self.resolved_n_modes_,
            "fit_total_modes": int(len(eigenvalues)),
            "eigenvalues_real": np.real(eigenvalues).tolist(),
            "eigenvalues_imag": np.imag(eigenvalues).tolist(),
            "mode_norms": mode_norms,
            "gram_condition_number": float(getattr(self.okhs, "gram_condition_number_", 0.0))
            if self.okhs is not None else None,
        }
