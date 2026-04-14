import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from fedot_ind.core.operation.decomposition.matrix_decomposition.dmd.dmd import FractionalDMD
from .okhs_common import (
    canonical_method_name,
    normalize_okhs_method,
    uses_dmd,
)
from .okhs_runtime import (
    analyze_okhs_smoothing_collapse as _analyze_okhs_smoothing_collapse_core,
    apply_okhs_anti_smoothing as _apply_okhs_anti_smoothing_core,
    build_dense_okhs_trajectories,
    build_okhs_dmd_model,
    build_okhs_direct_model,
    build_okhs_fit_plan,
    build_okhs_optimization_info,
    build_okhs_prediction_plan,
    execute_okhs_dmd_prediction_plan,
    postprocess_okhs_dmd_forecast,
    resolve_okhs_last_trajectory,
    resolve_projected_okhs_initial_state,
    run_okhs_direct_prediction,
    uses_projected_okhs_representation,
)
from ...operation.transformation.representation.kernel.kernels import OccupationKernel


def _resolve_anti_smoothing_tail_window(
        series_length: int,
        forecast_horizon: int,
        requested_window: int,
) -> int:
    if requested_window is not None:
        return int(max(forecast_horizon + 2, min(int(requested_window), series_length)))
    default_window = max(forecast_horizon * 2, 12)
    return int(max(forecast_horizon + 2, min(default_window, series_length)))


def _monotone_ratio(values: np.ndarray) -> float:
    normalized = np.asarray(values, dtype=float).reshape(-1)
    if normalized.size <= 2:
        return 1.0
    diffs = np.diff(normalized)
    significant = diffs[np.abs(diffs) > 1e-8]
    if significant.size <= 1:
        return 1.0
    signs = np.sign(significant)
    return float(np.max(np.bincount((signs > 0).astype(int))) / max(1, len(signs)))


def analyze_okhs_smoothing_collapse(
        train_series: np.ndarray,
        forecast: np.ndarray,
        *,
        forecast_horizon: int,
        tail_window: int = None,
        amplitude_ratio_threshold: float = 0.35,
        monotone_ratio_threshold: float = 0.9,
        oscillation_floor: float = 0.25,
) -> dict:
    return _analyze_okhs_smoothing_collapse_core(
        train_series=train_series,
        forecast=forecast,
        forecast_horizon=forecast_horizon,
        tail_window=tail_window,
        amplitude_ratio_threshold=amplitude_ratio_threshold,
        monotone_ratio_threshold=monotone_ratio_threshold,
        oscillation_floor=oscillation_floor,
    )


def apply_okhs_anti_smoothing(
        train_series: np.ndarray,
        forecast: np.ndarray,
        *,
        forecast_horizon: int,
        policy: str = 'residual_bridge',
        tail_window: int = None,
        amplitude_ratio_threshold: float = 0.35,
        monotone_ratio_threshold: float = 0.9,
        oscillation_floor: float = 0.25,
        decay: float = 2.5,
        target_amplitude_ratio: float = 0.8,
) -> tuple:
    return _apply_okhs_anti_smoothing_core(
        train_series=train_series,
        forecast=forecast,
        forecast_horizon=forecast_horizon,
        policy=policy,
        tail_window=tail_window,
        amplitude_ratio_threshold=amplitude_ratio_threshold,
        monotone_ratio_threshold=monotone_ratio_threshold,
        oscillation_floor=oscillation_floor,
        decay=decay,
        target_amplitude_ratio=target_amplitude_ratio,
    )


class OKHSForecaster(BaseEstimator, RegressorMixin):
    """
    Forecasting with Occupation Kernel Hilbert Spaces and fractional Liouville operators.
    """

    def __init__(
            self, 
            q=0.7,
            forecast_horizon=10,
            n_modes=5,
            method='dmd',
            forecasting_strategy='recursive',
            q_policy='fixed',
            q_selector=None,
            window_policy='adaptive_cycle_aware',
            trajectory_sampling_policy='dense',
            trajectory_rank_policy='explained_dispersion',
            trajectory_rank_value=None,
            trajectory_representation_policy='projected',
            latent_trajectory_stride_policy='adaptive',
            latent_trajectory_stride=None,
            mode_selection_policy='fixed',
            mode_energy_threshold=0.95,
            prediction_mode_selection_policy='adaptive_tail_energy',
            max_prediction_modes=None,
            min_prediction_modes=4,
            boundary_alignment_policy='tapered_offset',
            boundary_alignment_decay=4.0,
            prediction_stability_threshold=0.03,
            anti_smoothing_policy='residual_bridge',
            anti_smoothing_tail_window=None,
            anti_smoothing_amplitude_ratio=0.35,
            anti_smoothing_monotone_ratio=0.9,
            anti_smoothing_oscillation_floor=0.25,
            anti_smoothing_decay=2.5,
            anti_smoothing_target_amplitude_ratio=0.8,
            device = 'cpu',
    ):
        self.q = q
        self.forecast_horizon = forecast_horizon
        self.n_modes = n_modes
        self.method = normalize_okhs_method(method)
        self.forecasting_strategy = forecasting_strategy
        self.q_policy = q_policy
        self.q_selector = q_selector
        self.window_policy = window_policy
        self.trajectory_sampling_policy = trajectory_sampling_policy
        self.trajectory_rank_policy = trajectory_rank_policy
        self.trajectory_rank_value = trajectory_rank_value
        self.trajectory_representation_policy = trajectory_representation_policy
        self.latent_trajectory_stride_policy = latent_trajectory_stride_policy
        self.latent_trajectory_stride = latent_trajectory_stride
        self.mode_selection_policy = mode_selection_policy
        self.mode_energy_threshold = mode_energy_threshold
        self.prediction_mode_selection_policy = prediction_mode_selection_policy
        self.max_prediction_modes = max_prediction_modes
        self.min_prediction_modes = min_prediction_modes
        self.boundary_alignment_policy = boundary_alignment_policy
        self.boundary_alignment_decay = boundary_alignment_decay
        self.prediction_stability_threshold = prediction_stability_threshold
        self.anti_smoothing_policy = anti_smoothing_policy
        self.anti_smoothing_tail_window = anti_smoothing_tail_window
        self.anti_smoothing_amplitude_ratio = anti_smoothing_amplitude_ratio
        self.anti_smoothing_monotone_ratio = anti_smoothing_monotone_ratio
        self.anti_smoothing_oscillation_floor = anti_smoothing_oscillation_floor
        self.anti_smoothing_decay = anti_smoothing_decay
        self.anti_smoothing_target_amplitude_ratio = anti_smoothing_target_amplitude_ratio

        self.model = None
        self.resolved_q_ = q
        self.resolved_window_size_ = None
        self.window_diagnostics_ = None
        self.trajectory_preprocessing_ = None
        self.projection_metadata_ = None
        self._projection_runtime_ = None
        self.dmd_prediction_diagnostics_ = None
        self.direct_fit_diagnostics_ = None
        self.direct_prediction_diagnostics_ = None
        self.method_name_ = canonical_method_name(self.method)
        self.device = device
        self.train_series_ = None

    def _create_trajectories(self, time_series, window_size):
        return build_dense_okhs_trajectories(time_series, window_size)

    def fit(self, time_series, window_size=20):
        fit_plan = build_okhs_fit_plan(
            time_series=time_series,
            window_size=window_size,
            method=self.method,
            forecast_horizon=self.forecast_horizon,
            q=self.q,
            q_policy=self.q_policy,
            q_selector=self.q_selector,
            window_policy=self.window_policy,
            trajectory_sampling_policy=self.trajectory_sampling_policy,
            trajectory_rank_policy=self.trajectory_rank_policy,
            trajectory_rank_value=self.trajectory_rank_value,
            trajectory_representation_policy=self.trajectory_representation_policy,
            latent_trajectory_stride_policy=self.latent_trajectory_stride_policy,
            latent_trajectory_stride=self.latent_trajectory_stride,
        )
        self.train_series_ = fit_plan['train_series']
        self.dmd_prediction_diagnostics_ = None
        self.direct_fit_diagnostics_ = None
        self.direct_prediction_diagnostics_ = None
        self.window_diagnostics_ = fit_plan['window_diagnostics']
        self.projection_metadata_ = fit_plan['projection_metadata']
        self._projection_runtime_ = fit_plan['projection_runtime']
        self.resolved_window_size_ = fit_plan['resolved_window_size']
        self.window_size_ = self.resolved_window_size_
        self.trajectories_ = fit_plan['trajectories']
        self.trajectory_preprocessing_ = fit_plan['trajectory_preprocessing']
        self.resolved_q_ = fit_plan['resolved_q']

        if uses_dmd(self.method):
            self.model = build_okhs_dmd_model(
                dmd_factory=FractionalDMD,
                resolved_q=self.resolved_q_,
                n_modes=self.n_modes,
                mode_selection_policy=self.mode_selection_policy,
                mode_energy_threshold=self.mode_energy_threshold,
                prediction_mode_selection_policy=self.prediction_mode_selection_policy,
                max_prediction_modes=self.max_prediction_modes,
                min_prediction_modes=self.min_prediction_modes,
                boundary_alignment_policy=self.boundary_alignment_policy,
                boundary_alignment_decay=self.boundary_alignment_decay,
                prediction_stability_threshold=self.prediction_stability_threshold,
                device=self.device,
            )
            self.model.fit(self.trajectories_)
        else:
            self._fit_direct_okhs(time_series)

        return self

    def _fit_direct_okhs(self, time_series):
        del time_series
        direct_model = build_okhs_direct_model(
            kernel_factory=OccupationKernel,
            resolved_q=self.resolved_q_,
            trajectories=self.trajectories_,
        )
        self.kernel_ = direct_model['kernel']
        self.gram_matrix_ = direct_model['gram_matrix']
        self.weights_ = direct_model['weights']
        self.direct_fit_diagnostics_ = direct_model['fit_diagnostics']

    def _uses_projected_representation(self):
        return uses_projected_okhs_representation(
            projection_metadata=self.projection_metadata_,
            projection_runtime=self._projection_runtime_,
        )

    def _resolve_projected_initial_trajectory(self, time_series=None):
        return resolve_projected_okhs_initial_state(
            time_series=time_series,
            window_size=self.window_size_,
            trajectory_preprocessing=self.trajectory_preprocessing_,
            projection_runtime=self._projection_runtime_,
        )

    def _predict_projected_dmd(self, time_series=None):
        prediction_plan = build_okhs_prediction_plan(
            trajectories=self.trajectories_,
            time_series=time_series,
            train_series=self.train_series_,
            window_size=self.window_size_,
            trajectory_preprocessing=self.trajectory_preprocessing_,
            projection_metadata=self.projection_metadata_,
            projection_runtime=self._projection_runtime_,
        )
        forecast, diagnostics = execute_okhs_dmd_prediction_plan(
            model=self.model,
            prediction_plan=prediction_plan,
            forecast_horizon=self.forecast_horizon,
            projection_runtime=self._projection_runtime_,
        )
        self.dmd_prediction_diagnostics_ = diagnostics
        return forecast

    def _postprocess_dmd_forecast(self, forecast, time_series=None):
        reference_series = self.train_series_ if time_series is None else np.asarray(time_series, dtype=float).reshape(
            -1)
        corrected, merged_diagnostics = postprocess_okhs_dmd_forecast(
            train_series=reference_series,
            forecast=np.asarray(forecast, dtype=float).reshape(-1),
            forecast_horizon=self.forecast_horizon,
            prediction_diagnostics=self.dmd_prediction_diagnostics_,
            anti_smoothing_policy=self.anti_smoothing_policy,
            anti_smoothing_tail_window=self.anti_smoothing_tail_window,
            anti_smoothing_amplitude_ratio=self.anti_smoothing_amplitude_ratio,
            anti_smoothing_monotone_ratio=self.anti_smoothing_monotone_ratio,
            anti_smoothing_oscillation_floor=self.anti_smoothing_oscillation_floor,
            anti_smoothing_decay=self.anti_smoothing_decay,
            anti_smoothing_target_amplitude_ratio=self.anti_smoothing_target_amplitude_ratio,
        )
        self.dmd_prediction_diagnostics_ = merged_diagnostics
        return corrected

    def predict(self, time_series=None):
        if uses_dmd(self.method):
            if self._uses_projected_representation():
                return self._postprocess_dmd_forecast(self._predict_projected_dmd(time_series), time_series)

            prediction_plan = build_okhs_prediction_plan(
                trajectories=self.trajectories_,
                time_series=time_series,
                train_series=self.train_series_,
                window_size=self.window_size_,
                trajectory_preprocessing=self.trajectory_preprocessing_,
                projection_metadata=self.projection_metadata_,
                projection_runtime=self._projection_runtime_,
            )
            predictions, diagnostics = execute_okhs_dmd_prediction_plan(
                model=self.model,
                prediction_plan=prediction_plan,
                forecast_horizon=self.forecast_horizon,
            )
            self.dmd_prediction_diagnostics_ = diagnostics
            return self._postprocess_dmd_forecast(np.asarray(predictions, dtype=float).reshape(-1), time_series)

        last_trajectory = resolve_okhs_last_trajectory(
            trajectories=self.trajectories_,
            time_series=time_series,
            window_size=self.window_size_,
        )
        return self._predict_direct(last_trajectory)

    def _predict_direct(self, last_trajectory):
        predictions, diagnostics = run_okhs_direct_prediction(
            kernel=self.kernel_,
            reference_trajectories=self.trajectories_[:-1],
            last_trajectory=last_trajectory,
            weights=self.weights_,
            forecast_horizon=self.forecast_horizon,
        )
        self.direct_prediction_diagnostics_ = diagnostics
        return predictions

    def get_optimization_info(self):
        return build_okhs_optimization_info(
            method_name=self.method_name_,
            resolved_q=self.resolved_q_,
            q_policy=self.q_policy,
            forecast_horizon=self.forecast_horizon,
            window_policy=self.window_policy,
            trajectory_sampling_policy=self.trajectory_sampling_policy,
            trajectory_rank_policy=self.trajectory_rank_policy,
            trajectory_rank_value=self.trajectory_rank_value,
            trajectory_representation_policy=self.trajectory_representation_policy,
            latent_trajectory_stride_policy=self.latent_trajectory_stride_policy,
            latent_trajectory_stride=self.latent_trajectory_stride,
            resolved_window_size=self.resolved_window_size_,
            window_diagnostics=self.window_diagnostics_,
            trajectory_preprocessing=self.trajectory_preprocessing_,
            projection_metadata=self.projection_metadata_,
            mode_selection_policy=self.mode_selection_policy,
            mode_energy_threshold=self.mode_energy_threshold,
            prediction_mode_selection_policy=self.prediction_mode_selection_policy,
            max_prediction_modes=self.max_prediction_modes,
            min_prediction_modes=self.min_prediction_modes,
            boundary_alignment_policy=self.boundary_alignment_policy,
            boundary_alignment_decay=self.boundary_alignment_decay,
            prediction_stability_threshold=self.prediction_stability_threshold,
            anti_smoothing_policy=self.anti_smoothing_policy,
            anti_smoothing_tail_window=self.anti_smoothing_tail_window,
            anti_smoothing_amplitude_ratio=self.anti_smoothing_amplitude_ratio,
            anti_smoothing_monotone_ratio=self.anti_smoothing_monotone_ratio,
            anti_smoothing_oscillation_floor=self.anti_smoothing_oscillation_floor,
            anti_smoothing_decay=self.anti_smoothing_decay,
            anti_smoothing_target_amplitude_ratio=self.anti_smoothing_target_amplitude_ratio,
            model=self.model,
            dmd_prediction_diagnostics=self.dmd_prediction_diagnostics_,
            direct_fit_diagnostics=self.direct_fit_diagnostics_,
            direct_prediction_diagnostics=self.direct_prediction_diagnostics_,
            weights=getattr(self, 'weights_', None),
        )
