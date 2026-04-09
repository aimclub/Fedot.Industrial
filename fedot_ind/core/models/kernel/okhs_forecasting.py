import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from fedot_ind.core.operation.decomposition.matrix_decomposition.dmd.dmd import FractionalDMD
from .okhs_common import (
    analyze_okhs_window_size,
    build_okhs_projected_state_sequence,
    build_okhs_trajectory_representation,
    canonical_method_name,
    normalize_okhs_method,
    resolve_okhs_q,
    uses_dmd,
)
from ...operation.transformation.representation.kernel.kernels import OccupationKernel


def _resolve_anti_smoothing_tail_window(
        series_length: int,
        forecast_horizon: int,
        requested_window: int | None,
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
        tail_window: int | None = None,
        amplitude_ratio_threshold: float = 0.35,
        monotone_ratio_threshold: float = 0.9,
        oscillation_floor: float = 0.25,
) -> dict[str, float | int | bool]:
    train = np.asarray(train_series, dtype=float).reshape(-1)
    predicted = np.asarray(forecast, dtype=float).reshape(-1)
    if train.size == 0 or predicted.size == 0:
        return {
            'tail_window': 0,
            'collapse_detected': False,
            'train_tail_amplitude': 0.0,
            'forecast_amplitude_before': 0.0,
            'forecast_monotone_ratio_before': 0.0,
            'train_tail_oscillation_score': 0.0,
            'envelope_ratio_before': 0.0,
        }

    resolved_tail_window = _resolve_anti_smoothing_tail_window(
        series_length=len(train),
        forecast_horizon=forecast_horizon,
        requested_window=tail_window,
    )
    tail = train[-resolved_tail_window:]
    forecast_slice = predicted[: min(len(predicted), resolved_tail_window)]
    tail_diffs = np.diff(tail)
    sign_changes = np.sum(np.sign(tail_diffs[1:]) != np.sign(tail_diffs[:-1])) if len(tail_diffs) > 1 else 0
    oscillation_score = float(sign_changes / max(1, len(tail_diffs) - 1))
    train_tail_amplitude = float(np.std(tail))
    forecast_amplitude = float(np.std(forecast_slice))
    envelope_ratio = float(forecast_amplitude / max(train_tail_amplitude, 1e-8))
    forecast_monotone_ratio = _monotone_ratio(forecast_slice)
    collapse_detected = (
            oscillation_score >= float(oscillation_floor)
            and forecast_monotone_ratio >= float(monotone_ratio_threshold)
            and envelope_ratio <= float(amplitude_ratio_threshold)
    )
    return {
        'tail_window': int(resolved_tail_window),
        'collapse_detected': bool(collapse_detected),
        'train_tail_amplitude': train_tail_amplitude,
        'forecast_amplitude_before': forecast_amplitude,
        'forecast_monotone_ratio_before': float(forecast_monotone_ratio),
        'train_tail_oscillation_score': oscillation_score,
        'envelope_ratio_before': envelope_ratio,
        'amplitude_ratio_threshold': float(amplitude_ratio_threshold),
        'monotone_ratio_threshold': float(monotone_ratio_threshold),
        'oscillation_floor': float(oscillation_floor),
    }


def apply_okhs_anti_smoothing(
        train_series: np.ndarray,
        forecast: np.ndarray,
        *,
        forecast_horizon: int,
        policy: str = 'residual_bridge',
        tail_window: int | None = None,
        amplitude_ratio_threshold: float = 0.35,
        monotone_ratio_threshold: float = 0.9,
        oscillation_floor: float = 0.25,
        decay: float = 2.5,
        target_amplitude_ratio: float = 0.8,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    predicted = np.asarray(forecast, dtype=float).reshape(-1)
    diagnostics = analyze_okhs_smoothing_collapse(
        train_series=train_series,
        forecast=predicted,
        forecast_horizon=forecast_horizon,
        tail_window=tail_window,
        amplitude_ratio_threshold=amplitude_ratio_threshold,
        monotone_ratio_threshold=monotone_ratio_threshold,
        oscillation_floor=oscillation_floor,
    )
    diagnostics = {
        'anti_smoothing_policy': str(policy),
        'correction_applied': False,
        'residual_scale': 0.0,
        'target_amplitude_ratio': float(target_amplitude_ratio),
        'forecast_amplitude_after': diagnostics['forecast_amplitude_before'],
        'forecast_monotone_ratio_after': diagnostics['forecast_monotone_ratio_before'],
        **diagnostics,
    }
    if str(policy).lower() == 'none' or not diagnostics['collapse_detected']:
        return predicted, diagnostics

    train = np.asarray(train_series, dtype=float).reshape(-1)
    resolved_tail_window = int(diagnostics['tail_window'])
    tail = train[-resolved_tail_window:]
    tail_index = np.arange(len(tail), dtype=float)
    slope, intercept = np.polyfit(tail_index, tail, deg=1)
    residual = tail - (slope * tail_index + intercept)
    residual = residual - np.mean(residual)
    if np.std(residual) <= 1e-8:
        return predicted, diagnostics

    repeats = int(np.ceil(len(predicted) / len(residual)))
    residual_template = np.tile(residual, repeats)[: len(predicted)]
    residual_template = residual_template - np.mean(residual_template)
    template_amplitude = float(np.std(residual_template))
    if template_amplitude <= 1e-8:
        return predicted, diagnostics

    target_amplitude = float(diagnostics['train_tail_amplitude']) * float(target_amplitude_ratio)
    amplitude_gap = max(0.0, target_amplitude - float(diagnostics['forecast_amplitude_before']))
    residual_scale = min(1.5, amplitude_gap / template_amplitude) if amplitude_gap > 0 else 0.0
    if residual_scale <= 0.0:
        return predicted, diagnostics

    weights = np.exp(-np.linspace(0.0, float(decay), len(predicted), dtype=float))
    corrected = predicted + residual_scale * weights * residual_template
    diagnostics.update(
        {
            'correction_applied': True,
            'residual_scale': float(residual_scale),
            'forecast_amplitude_after': float(np.std(corrected[: min(len(corrected), resolved_tail_window)])),
            'forecast_monotone_ratio_after': float(
                _monotone_ratio(corrected[: min(len(corrected), resolved_tail_window)])
            ),
            'residual_template_amplitude': template_amplitude,
            'residual_template_tail': residual_template[: min(len(residual_template), 8)].tolist(),
            'correction_weights_head': weights[: min(len(weights), 8)].tolist(),
        }
    )
    return corrected, diagnostics


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
        self.method_name_ = canonical_method_name(self.method)
        self.device = device
        self.train_series_ = None

    def _create_trajectories(self, time_series, window_size):
        trajectories = []
        for index in range(len(time_series) - window_size):
            trajectories.append(time_series[index:index + window_size])
        return trajectories

    def fit(self, time_series, window_size=20):
        self.train_series_ = np.asarray(time_series, dtype=float).reshape(-1)
        self.window_diagnostics_ = analyze_okhs_window_size(
            window_size=window_size,
            window_policy=self.window_policy,
            time_series=time_series,
            forecast_horizon=self.forecast_horizon,
        )
        self.dmd_prediction_diagnostics_ = None
        self.projection_metadata_ = None
        self._projection_runtime_ = None
        self.resolved_window_size_ = self.window_diagnostics_["resolved_window_size"]
        self.window_size_ = self.resolved_window_size_

        if uses_dmd(self.method):
            representation = build_okhs_trajectory_representation(
                time_series=time_series,
                window_size=self.window_size_,
                window_policy=self.window_policy,
                forecast_horizon=self.forecast_horizon,
                trajectory_sampling_policy=self.trajectory_sampling_policy,
                trajectory_rank_policy=self.trajectory_rank_policy,
                trajectory_rank_value=self.trajectory_rank_value,
                trajectory_representation_policy=self.trajectory_representation_policy,
                latent_trajectory_stride_policy=self.latent_trajectory_stride_policy,
                latent_trajectory_stride=self.latent_trajectory_stride,
            )
            self.trajectories_ = representation["training_matrix"]
            self.trajectory_preprocessing_ = representation["trajectory_preprocessing"]
            self.projection_metadata_ = representation["projection_metadata"]
            self._projection_runtime_ = representation["projection_runtime"]
        else:
            self.trajectories_ = self._create_trajectories(time_series, self.window_size_)
            self.trajectory_preprocessing_ = None

        self.resolved_q_ = resolve_okhs_q(
            q=self.q,
            q_policy=self.q_policy,
            trajectories=self.trajectories_,
            q_selector=self.q_selector,
        )

        if uses_dmd(self.method):
            dmd_kwargs = {
                'q': self.resolved_q_,
                'n_modes': self.n_modes,
                'mode_selection_policy': self.mode_selection_policy,
                'mode_energy_threshold': self.mode_energy_threshold,
                'prediction_mode_selection_policy': self.prediction_mode_selection_policy,
                'max_prediction_modes': self.max_prediction_modes,
                'min_prediction_modes': self.min_prediction_modes,
                'boundary_alignment_policy': self.boundary_alignment_policy,
                'boundary_alignment_decay': self.boundary_alignment_decay,
                'prediction_stability_threshold': self.prediction_stability_threshold,
                'device': self.device,
            }
            try:
                self.model = FractionalDMD(**dmd_kwargs)
            except TypeError:
                dmd_kwargs.pop('device', None)
                self.model = FractionalDMD(**dmd_kwargs)
            self.model.fit(self.trajectories_)
        else:
            self._fit_direct_okhs(time_series)

        return self

    def _fit_direct_okhs(self, time_series):
        del time_series
        self.kernel_ = OccupationKernel(q=self.resolved_q_)
        x_trajectories = self.trajectories_[:-1]
        y_targets = [traj[-1] for traj in self.trajectories_[1:]]
        self.gram_matrix_ = self.kernel_.compute_gram_matrix(x_trajectories)
        self.weights_ = np.linalg.lstsq(self.gram_matrix_, y_targets, rcond=None)[0]

    def _uses_projected_representation(self):
        return (
                self.projection_metadata_ is not None
                and self.projection_metadata_.get('representation_policy') == 'projected'
                and self.projection_metadata_.get('decode_supported') is True
                and self._projection_runtime_ is not None
        )

    def _resolve_projected_initial_trajectory(self, time_series=None):
        runtime = self._projection_runtime_ or {}
        latent_window_size = int(runtime.get("latent_window_size", 0))
        if latent_window_size <= 0:
            raise ValueError("Projected OKHS DMD path requires latent_window_size.")

        if time_series is None:
            latent_states = np.asarray(runtime["latent_state_matrix"], dtype=float)
            sampled_matrix = np.asarray(runtime["sampled_matrix"], dtype=float)
            return latent_states[-latent_window_size:], sampled_matrix[-latent_window_size:]

        sampled_matrix, latent_states = build_okhs_projected_state_sequence(
            time_series=time_series,
            window_size=self.window_size_,
            effective_stride=int(self.trajectory_preprocessing_["effective_stride"]),
            basis=np.asarray(runtime["basis"], dtype=float),
        )
        if latent_states.shape[0] < latent_window_size:
            raise ValueError(
                "Insufficient projected latent states for prediction: "
                f"{latent_states.shape[0]} < {latent_window_size}."
            )
        return latent_states[-latent_window_size:], sampled_matrix[-latent_window_size:]

    def _predict_projected_dmd(self, time_series=None):
        initial_trajectory, decoded_initial_trajectory = self._resolve_projected_initial_trajectory(time_series)
        time_step = float(getattr(self.model, 'dt', 1.0))
        start_time = len(initial_trajectory) * time_step
        stop_time = start_time + self.forecast_horizon * time_step
        future_times = np.arange(start_time, stop_time, time_step)

        if hasattr(self.model, "predict_with_diagnostics"):
            latent_prediction, diagnostics = self.model.predict_with_diagnostics(initial_trajectory, future_times)
        else:
            latent_prediction = self.model.predict(initial_trajectory, future_times)
            diagnostics = {}

        latent_prediction = np.asarray(latent_prediction, dtype=float)
        if latent_prediction.ndim == 1:
            latent_prediction = latent_prediction.reshape(-1, 1)

        basis = np.asarray(self._projection_runtime_["basis"], dtype=float)
        decoded_prediction = latent_prediction @ basis.T
        forecast = decoded_prediction[:, -1].reshape(-1)
        decoded_initial = np.asarray(initial_trajectory, dtype=float) @ basis.T
        decode_reconstruction_error = float(
            np.sqrt(np.mean((decoded_initial - np.asarray(decoded_initial_trajectory, dtype=float)) ** 2))
        )
        diagnostics = {
            **diagnostics,
            "representation_policy": "projected",
            "basis_shape": tuple(int(value) for value in basis.shape),
            "projected_shape": tuple(
                int(value) for value in np.asarray(self._projection_runtime_["latent_state_matrix"]).shape
            ),
            "decode_supported": True,
            "latent_prediction_shape": tuple(int(value) for value in latent_prediction.shape),
            "decoded_prediction_shape": tuple(int(value) for value in decoded_prediction.shape),
            "latent_first_prediction_value": latent_prediction[0].tolist() if len(latent_prediction) else None,
            "decoded_first_prediction_window": decoded_prediction[0].tolist() if len(decoded_prediction) else None,
            "decoded_first_prediction_value": float(forecast[0]) if len(forecast) else None,
            "decode_reconstruction_error": decode_reconstruction_error,
        }
        self.dmd_prediction_diagnostics_ = diagnostics
        return forecast

    def _postprocess_dmd_forecast(self, forecast, time_series=None):
        reference_series = self.train_series_ if time_series is None else np.asarray(time_series, dtype=float).reshape(
            -1)
        corrected, anti_smoothing_diagnostics = apply_okhs_anti_smoothing(
            train_series=reference_series,
            forecast=np.asarray(forecast, dtype=float).reshape(-1),
            forecast_horizon=self.forecast_horizon,
            policy=self.anti_smoothing_policy,
            tail_window=self.anti_smoothing_tail_window,
            amplitude_ratio_threshold=self.anti_smoothing_amplitude_ratio,
            monotone_ratio_threshold=self.anti_smoothing_monotone_ratio,
            oscillation_floor=self.anti_smoothing_oscillation_floor,
            decay=self.anti_smoothing_decay,
            target_amplitude_ratio=self.anti_smoothing_target_amplitude_ratio,
        )
        if self.dmd_prediction_diagnostics_ is None:
            self.dmd_prediction_diagnostics_ = {}
        self.dmd_prediction_diagnostics_ = {
            **self.dmd_prediction_diagnostics_,
            'anti_smoothing_diagnostics': anti_smoothing_diagnostics,
        }
        return corrected

    def predict(self, time_series=None):
        if uses_dmd(self.method):
            if self._uses_projected_representation():
                return self._postprocess_dmd_forecast(self._predict_projected_dmd(time_series), time_series)

            if time_series is None:
                last_trajectory = self.trajectories_[-1]
            else:
                last_trajectory = time_series[-self.window_size_:]

            time_step = float(getattr(self.model, 'dt', 1.0))
            start_time = len(last_trajectory) * time_step
            stop_time = start_time + self.forecast_horizon * time_step
            future_times = np.arange(start_time, stop_time, time_step)
            if hasattr(self.model, "predict_with_diagnostics"):
                predictions, diagnostics = self.model.predict_with_diagnostics(last_trajectory, future_times)
                self.dmd_prediction_diagnostics_ = diagnostics
            else:
                predictions = self.model.predict(last_trajectory, future_times)
            # Ensure predictions are on CPU before converting to numpy
            if hasattr(predictions, 'cpu'):
                predictions = predictions.cpu()
            return self._postprocess_dmd_forecast(np.asarray(predictions, dtype=float).flatten(), time_series)

        if time_series is None:
            last_trajectory = self.trajectories_[-1]
        else:
            last_trajectory = time_series[-self.window_size_:]
        return self._predict_direct(last_trajectory)

    def _predict_direct(self, last_trajectory):
        predictions = []
        current_trajectory = last_trajectory.copy()

        for _ in range(self.forecast_horizon):
            kernels = []
            for train_traj in self.trajectories_[:-1]:
                kernel_val = self.kernel_._compute_trajectory_kernel(current_trajectory, train_traj)
                kernels.append(kernel_val)

            kernels = np.array(kernels)
            prediction = kernels @ self.weights_
            predictions.append(prediction)
            current_trajectory = np.roll(current_trajectory, -1)
            current_trajectory[-1] = prediction

        return np.array(predictions)

    def get_optimization_info(self):
        info = {
            'method': self.method_name_,
            'q': self.resolved_q_,
            'q_policy': self.q_policy,
            'forecast_horizon': self.forecast_horizon,
            'window_policy': self.window_policy,
            'trajectory_sampling_policy': self.trajectory_sampling_policy,
            'trajectory_rank_policy': self.trajectory_rank_policy,
            'trajectory_rank_value': self.trajectory_rank_value,
            'trajectory_representation_policy': self.trajectory_representation_policy,
            'latent_trajectory_stride_policy': self.latent_trajectory_stride_policy,
            'latent_trajectory_stride': self.latent_trajectory_stride,
            'resolved_window_size': self.resolved_window_size_,
            'window_diagnostics': self.window_diagnostics_,
            'trajectory_preprocessing': self.trajectory_preprocessing_,
            'projection_metadata': self.projection_metadata_,
            'mode_selection_policy': self.mode_selection_policy,
            'mode_energy_threshold': self.mode_energy_threshold,
            'prediction_mode_selection_policy': self.prediction_mode_selection_policy,
            'max_prediction_modes': self.max_prediction_modes,
            'min_prediction_modes': self.min_prediction_modes,
            'boundary_alignment_policy': self.boundary_alignment_policy,
            'boundary_alignment_decay': self.boundary_alignment_decay,
            'prediction_stability_threshold': self.prediction_stability_threshold,
            'anti_smoothing_policy': self.anti_smoothing_policy,
            'anti_smoothing_tail_window': self.anti_smoothing_tail_window,
            'anti_smoothing_amplitude_ratio': self.anti_smoothing_amplitude_ratio,
            'anti_smoothing_monotone_ratio': self.anti_smoothing_monotone_ratio,
            'anti_smoothing_oscillation_floor': self.anti_smoothing_oscillation_floor,
            'anti_smoothing_decay': self.anti_smoothing_decay,
            'anti_smoothing_target_amplitude_ratio': self.anti_smoothing_target_amplitude_ratio,
        }
        if self.model is not None and hasattr(self.model, 'get_fit_diagnostics_summary'):
            info['fdmd_fit_diagnostics'] = self.model.get_fit_diagnostics_summary()
        if self.dmd_prediction_diagnostics_ is not None:
            info['fdmd_prediction_diagnostics'] = self.dmd_prediction_diagnostics_
        if hasattr(self, 'weights_'):
            info['weights_norm'] = float(np.linalg.norm(self.weights_))
        return info
