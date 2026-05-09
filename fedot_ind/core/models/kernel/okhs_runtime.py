from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .okhs_common import (
    analyze_okhs_window_size,
    build_okhs_projected_state_sequence,
    build_okhs_trajectory_representation,
    normalize_okhs_method,
    resolve_okhs_q,
    uses_dmd,
)


def build_dense_okhs_trajectories(time_series: Sequence[float], window_size: int) -> list[np.ndarray]:
    series = np.asarray(time_series, dtype=float).reshape(-1)
    return [
        series[index:index + int(window_size)]
        for index in range(max(0, len(series) - int(window_size)))
    ]


def build_okhs_fit_plan(
        *,
        time_series: Sequence[float],
        window_size: int,
        method: str,
        forecast_horizon: int,
        q: float,
        q_policy: str,
        q_selector: Any = None,
        window_policy: str = 'adaptive_cycle_aware',
        trajectory_sampling_policy: str = 'dense',
        trajectory_rank_policy: str = 'explained_dispersion',
        trajectory_rank_value: int | None = None,
        trajectory_representation_policy: str = 'projected',
        latent_trajectory_stride_policy: str = 'adaptive',
        latent_trajectory_stride: int | None = None,
) -> dict[str, Any]:
    series = np.asarray(time_series, dtype=float).reshape(-1)
    normalized_method = normalize_okhs_method(method)
    window_diagnostics = analyze_okhs_window_size(
        window_size=window_size,
        window_policy=window_policy,
        time_series=series,
        forecast_horizon=forecast_horizon,
    )
    resolved_window_size = int(window_diagnostics['resolved_window_size'])

    if uses_dmd(normalized_method):
        representation = build_okhs_trajectory_representation(
            time_series=series,
            window_size=resolved_window_size,
            window_policy=window_policy,
            forecast_horizon=forecast_horizon,
            trajectory_sampling_policy=trajectory_sampling_policy,
            trajectory_rank_policy=trajectory_rank_policy,
            trajectory_rank_value=trajectory_rank_value,
            trajectory_representation_policy=trajectory_representation_policy,
            latent_trajectory_stride_policy=latent_trajectory_stride_policy,
            latent_trajectory_stride=latent_trajectory_stride,
        )
        trajectories = representation['training_matrix']
        trajectory_preprocessing = representation['trajectory_preprocessing']
        projection_metadata = representation['projection_metadata']
        projection_runtime = representation['projection_runtime']
    else:
        trajectories = build_dense_okhs_trajectories(series, resolved_window_size)
        trajectory_preprocessing = None
        projection_metadata = None
        projection_runtime = None

    resolved_q = resolve_okhs_q(
        q=q,
        q_policy=q_policy,
        trajectories=trajectories,
        q_selector=q_selector,
    )

    return {
        'train_series': series,
        'method': normalized_method,
        'window_diagnostics': window_diagnostics,
        'resolved_window_size': resolved_window_size,
        'trajectories': trajectories,
        'trajectory_preprocessing': trajectory_preprocessing,
        'projection_metadata': projection_metadata,
        'projection_runtime': projection_runtime,
        'resolved_q': resolved_q,
    }


def _safe_tuple(value: Any):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(int(item) for item in value)
    if isinstance(value, list):
        return tuple(int(item) for item in value)
    return value


def build_okhs_stage_diagnostics(optimization_info: dict[str, Any]) -> dict[str, Any]:
    window_diagnostics = dict(optimization_info.get('window_diagnostics') or {})
    preprocessing = dict(optimization_info.get('trajectory_preprocessing') or {})
    projection = dict(optimization_info.get('projection_metadata') or {})
    fit_diagnostics = dict(
        optimization_info.get('fdmd_fit_diagnostics')
        or optimization_info.get('direct_fit_diagnostics')
        or {}
    )
    prediction_diagnostics = dict(
        optimization_info.get('fdmd_prediction_diagnostics')
        or optimization_info.get('direct_prediction_diagnostics')
        or {}
    )
    anti_smoothing = dict(prediction_diagnostics.get('anti_smoothing_diagnostics') or {})

    trajectory_transform = {
        'window_policy': optimization_info.get('window_policy'),
        'resolved_window_size': optimization_info.get('resolved_window_size'),
        'window_fraction': window_diagnostics.get('window_fraction'),
        'expected_overlap_ratio': preprocessing.get(
            'expected_overlap_ratio',
            window_diagnostics.get('expected_overlap_ratio'),
        ),
        'effective_stride': preprocessing.get('effective_stride'),
        'dense_trajectory_count': preprocessing.get('dense_trajectory_count'),
        'effective_trajectory_count': preprocessing.get('effective_trajectory_count'),
        'trajectory_matrix_shape_before': _safe_tuple(preprocessing.get('trajectory_matrix_shape_before')),
        'trajectory_matrix_shape_after': _safe_tuple(preprocessing.get('trajectory_matrix_shape_after')),
    }

    decomposition = {
        'representation_policy': optimization_info.get('trajectory_representation_policy'),
        'projected_shape': _safe_tuple(projection.get('projected_shape')),
        'basis_shape': _safe_tuple(projection.get('basis_shape')),
        'decode_supported': bool(projection.get('decode_supported', False)),
        'decode_reconstruction_error': projection.get('decode_reconstruction_error'),
        'latent_window_size': projection.get('latent_window_size'),
        'latent_stride': projection.get('latent_stride'),
        'latent_overlap_ratio': projection.get('latent_overlap_ratio'),
    }

    rank_truncation = {
        'trajectory_rank_policy': optimization_info.get('trajectory_rank_policy'),
        'selected_rank': preprocessing.get('selected_rank'),
        'raw_selected_rank': preprocessing.get('raw_selected_rank'),
        'requested_rank_floor': preprocessing.get('requested_rank_floor'),
        'applied_rank_floor': preprocessing.get('applied_rank_floor'),
        'rank_floor_applied': preprocessing.get('rank_floor_applied'),
        'explained_variance_retained': preprocessing.get('explained_variance_retained'),
        'compression_ratio': preprocessing.get('compression_ratio'),
    }

    forecast_head = {
        'q': optimization_info.get('q'),
        'q_policy': optimization_info.get('q_policy'),
        'head_runtime': optimization_info.get('head_runtime'),
        'mode_selection_policy': optimization_info.get('mode_selection_policy'),
        'mode_energy_threshold': optimization_info.get('mode_energy_threshold'),
        'prediction_mode_selection_policy': optimization_info.get('prediction_mode_selection_policy'),
        'max_prediction_modes': optimization_info.get('max_prediction_modes'),
        'min_prediction_modes': optimization_info.get('min_prediction_modes'),
        'boundary_alignment_policy': optimization_info.get('boundary_alignment_policy'),
        'boundary_alignment_decay': optimization_info.get('boundary_alignment_decay'),
        'prediction_stability_threshold': optimization_info.get('prediction_stability_threshold'),
        'fit_diagnostics': fit_diagnostics,
        'prediction_diagnostics': prediction_diagnostics,
        'anti_smoothing': anti_smoothing,
    }

    return {
        'trajectory_transform': trajectory_transform,
        'decomposition': decomposition,
        'rank_truncation': rank_truncation,
        'forecast_head': forecast_head,
    }


def build_okhs_dmd_model(
        *,
        dmd_factory: Any,
        resolved_q: float,
        n_modes: int,
        mode_selection_policy: str,
        mode_energy_threshold: float,
        prediction_mode_selection_policy: str,
        max_prediction_modes: int | None,
        min_prediction_modes: int,
        boundary_alignment_policy: str,
        boundary_alignment_decay: float,
        prediction_stability_threshold: float | None,
        device: str = 'cpu',
):
    dmd_kwargs = {
        'q': resolved_q,
        'n_modes': n_modes,
        'mode_selection_policy': mode_selection_policy,
        'mode_energy_threshold': mode_energy_threshold,
        'prediction_mode_selection_policy': prediction_mode_selection_policy,
        'max_prediction_modes': max_prediction_modes,
        'min_prediction_modes': min_prediction_modes,
        'boundary_alignment_policy': boundary_alignment_policy,
        'boundary_alignment_decay': boundary_alignment_decay,
        'prediction_stability_threshold': prediction_stability_threshold,
        'device': device,
    }
    try:
        return dmd_factory(**dmd_kwargs)
    except TypeError:
        dmd_kwargs.pop('device', None)
        return dmd_factory(**dmd_kwargs)


def build_okhs_direct_model(
        *,
        kernel_factory: Any,
        resolved_q: float,
        trajectories: Sequence[np.ndarray],
) -> dict[str, Any]:
    kernel = kernel_factory(q=resolved_q)
    x_trajectories = [np.asarray(traj, dtype=float) for traj in trajectories[:-1]]
    y_targets = [float(np.asarray(traj, dtype=float).reshape(-1)[-1]) for traj in trajectories[1:]]
    gram_matrix = kernel.compute_gram_matrix(x_trajectories)
    weights = np.linalg.lstsq(gram_matrix, y_targets, rcond=None)[0]
    return {
        'kernel': kernel,
        'gram_matrix': np.asarray(gram_matrix, dtype=float),
        'weights': np.asarray(weights, dtype=float),
        'fit_diagnostics': {
            'n_reference_trajectories': int(len(x_trajectories)),
            'gram_matrix_shape': tuple(int(value) for value in np.asarray(gram_matrix).shape),
            'weights_shape': tuple(int(value) for value in np.asarray(weights).shape),
        },
    }


def  run_okhs_direct_prediction(
        *,
        kernel: Any,
        reference_trajectories: Sequence[np.ndarray],
        last_trajectory: np.ndarray,
        weights: np.ndarray,
        forecast_horizon: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    predictions: list[float] = []
    current_trajectory = np.asarray(last_trajectory, dtype=float).reshape(-1).copy()
    normalized_weights = np.asarray(weights, dtype=float).reshape(-1)
    normalized_references = [np.asarray(traj, dtype=float).reshape(-1) for traj in reference_trajectories]

    for _ in range(int(forecast_horizon)):
        kernels = np.asarray(
            [kernel._compute_trajectory_kernel(current_trajectory, train_traj) for train_traj in normalized_references],
            dtype=float,
        )
        prediction = float(kernels @ normalized_weights)
        predictions.append(prediction)
        current_trajectory = np.roll(current_trajectory, -1)
        current_trajectory[-1] = prediction

    return np.asarray(predictions, dtype=float), {
        'reference_trajectory_count': int(len(normalized_references)),
        'forecast_horizon': int(forecast_horizon),
        'last_trajectory_length': int(len(np.asarray(last_trajectory).reshape(-1))),
        'weights_norm': float(np.linalg.norm(normalized_weights)),
    }


def resolve_okhs_prediction_time_grid(initial_trajectory_length: int, forecast_horizon: int,
                                      time_step: float) -> np.ndarray:
    start_time = float(initial_trajectory_length) * float(time_step)
    stop_time = start_time + int(forecast_horizon) * float(time_step)
    return np.arange(start_time, stop_time, float(time_step))


def resolve_okhs_last_trajectory(
        trajectories: Sequence[np.ndarray],
        time_series: Sequence[float] | None,
        window_size: int,
) -> np.ndarray:
    if time_series is None:
        return np.asarray(trajectories[-1], dtype=float)
    series = np.asarray(time_series, dtype=float).reshape(-1)
    return series[-int(window_size):]


def uses_projected_okhs_representation(projection_metadata: dict[str, Any] | None,
                                       projection_runtime: dict[str, Any] | None) -> bool:
    return (
        projection_metadata is not None
        and projection_metadata.get('representation_policy') == 'projected'
        and projection_metadata.get('decode_supported') is True
        and projection_runtime is not None
    )


def build_okhs_prediction_plan(
        *,
        trajectories: Sequence[np.ndarray],
        time_series: Sequence[float] | None,
        train_series: Sequence[float],
        window_size: int,
        trajectory_preprocessing: dict[str, Any] | None,
        projection_metadata: dict[str, Any] | None,
        projection_runtime: dict[str, Any] | None,
) -> dict[str, Any]:
    reference_series = np.asarray(
        train_series if time_series is None else time_series,
        dtype=float,
    ).reshape(-1)
    if uses_projected_okhs_representation(projection_metadata, projection_runtime):
        initial_trajectory, decoded_initial_trajectory = resolve_projected_okhs_initial_state(
            time_series=time_series,
            window_size=window_size,
            trajectory_preprocessing=trajectory_preprocessing or {},
            projection_runtime=projection_runtime or {},
        )
        return {
            'representation_policy': 'projected',
            'initial_trajectory': np.asarray(initial_trajectory, dtype=float),
            'decoded_initial_trajectory': np.asarray(decoded_initial_trajectory, dtype=float),
            'reference_series': reference_series,
        }

    last_trajectory = resolve_okhs_last_trajectory(
        trajectories=trajectories,
        time_series=time_series,
        window_size=window_size,
    )
    return {
        'representation_policy': 'dense',
        'initial_trajectory': np.asarray(last_trajectory, dtype=float),
        'decoded_initial_trajectory': None,
        'reference_series': reference_series,
    }


def resolve_projected_okhs_initial_state(
        *,
        time_series: Sequence[float] | None,
        window_size: int,
        trajectory_preprocessing: dict[str, Any],
        projection_runtime: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    runtime = projection_runtime or {}
    latent_window_size = int(runtime.get('latent_window_size', 0))
    if latent_window_size <= 0:
        raise ValueError('Projected OKHS DMD path requires latent_window_size.')

    if time_series is None:
        latent_states = np.asarray(runtime['latent_state_matrix'], dtype=float)
        sampled_matrix = np.asarray(runtime['sampled_matrix'], dtype=float)
        return latent_states[-latent_window_size:], sampled_matrix[-latent_window_size:]

    sampled_matrix, latent_states = build_okhs_projected_state_sequence(
        time_series=time_series,
        window_size=window_size,
        effective_stride=int(trajectory_preprocessing['effective_stride']),
        basis=np.asarray(runtime['basis'], dtype=float),
    )
    if latent_states.shape[0] < latent_window_size:
        raise ValueError(
            'Insufficient projected latent states for prediction: '
            f'{latent_states.shape[0]} < {latent_window_size}.'
        )
    return latent_states[-latent_window_size:], sampled_matrix[-latent_window_size:]


def run_okhs_dmd_prediction(model: Any, initial_trajectory: np.ndarray, forecast_horizon: int) -> tuple[
        np.ndarray, dict[str, Any]]:
    time_step = float(getattr(model, 'dt', 1.0))
    future_times = resolve_okhs_prediction_time_grid(
        initial_trajectory_length=len(initial_trajectory),
        forecast_horizon=forecast_horizon,
        time_step=time_step,
    )
    # if hasattr(model, 'predict_with_diagnostics'):
    #     prediction, diagnostics = model.predict_with_diagnostics(initial_trajectory, future_times)
    # else:
    prediction = model.plot_predict(initial_trajectory, future_times)
    diagnostics = {}
    if hasattr(prediction, 'cpu'):
        prediction = prediction.cpu()
    normalized = np.asarray(prediction, dtype=float)
    return normalized, {**dict(diagnostics), 'prediction_time_grid': np.asarray(future_times, dtype=float).tolist()}


def decode_okhs_projected_prediction(
        *,
        initial_trajectory: np.ndarray,
        decoded_initial_trajectory: np.ndarray,
        latent_prediction: np.ndarray,
        projection_runtime: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    normalized_prediction = np.asarray(latent_prediction, dtype=float)
    if normalized_prediction.ndim == 1:
        normalized_prediction = normalized_prediction.reshape(-1, 1)

    basis = np.asarray(projection_runtime['basis'], dtype=float)
    decoded_prediction = normalized_prediction @ basis.T
    forecast = decoded_prediction[:, -1].reshape(-1)
    decoded_initial = np.asarray(initial_trajectory, dtype=float) @ basis.T
    decode_reconstruction_error = float(
        np.sqrt(np.mean((decoded_initial - np.asarray(decoded_initial_trajectory, dtype=float)) ** 2))
    )
    return forecast, {
        'representation_policy': 'projected',
        'basis_shape': tuple(int(value) for value in basis.shape),
        'projected_shape': tuple(int(value) for value in np.asarray(projection_runtime['latent_state_matrix']).shape),
        'decode_supported': True,
        'latent_prediction_shape': tuple(int(value) for value in normalized_prediction.shape),
        'decoded_prediction_shape': tuple(int(value) for value in decoded_prediction.shape),
        'latent_first_prediction_value': normalized_prediction[0].tolist() if len(normalized_prediction) else None,
        'decoded_first_prediction_window': decoded_prediction[0].tolist() if len(decoded_prediction) else None,
        'decoded_first_prediction_value': float(forecast[0]) if len(forecast) else None,
        'decode_reconstruction_error': decode_reconstruction_error,
    }


def execute_okhs_dmd_prediction_plan(
        *,
        model: Any,
        prediction_plan: dict[str, Any],
        forecast_horizon: int,
        projection_runtime: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    initial_trajectory = np.asarray(prediction_plan['initial_trajectory'], dtype=float)
    prediction, diagnostics = run_okhs_dmd_prediction(
        model=model,
        initial_trajectory=initial_trajectory,
        forecast_horizon=forecast_horizon,
    )
    if prediction_plan.get('representation_policy') == 'projected':
        forecast, decode_diagnostics = decode_okhs_projected_prediction(
            initial_trajectory=initial_trajectory,
            decoded_initial_trajectory=np.asarray(prediction_plan['decoded_initial_trajectory'], dtype=float),
            latent_prediction=prediction,
            projection_runtime=projection_runtime or {},
        )
        return forecast, {**diagnostics, **decode_diagnostics}
    return np.asarray(prediction, dtype=float).reshape(-1), diagnostics


def build_okhs_optimization_info(
        *,
        method_name: str,
        resolved_q: float,
        q_policy: str,
        forecast_horizon: int,
        window_policy: str,
        trajectory_sampling_policy: str,
        trajectory_rank_policy: str,
        trajectory_rank_value: int | None,
        trajectory_representation_policy: str,
        latent_trajectory_stride_policy: str,
        latent_trajectory_stride: int | None,
        resolved_window_size: int | None,
        window_diagnostics: dict[str, Any] | None,
        trajectory_preprocessing: dict[str, Any] | None,
        projection_metadata: dict[str, Any] | None,
        mode_selection_policy: str,
        mode_energy_threshold: float,
        prediction_mode_selection_policy: str,
        max_prediction_modes: int | None,
        min_prediction_modes: int,
        boundary_alignment_policy: str,
        boundary_alignment_decay: float,
        prediction_stability_threshold: float | None,
        anti_smoothing_policy: str,
        anti_smoothing_tail_window: int | None,
        anti_smoothing_amplitude_ratio: float,
        anti_smoothing_monotone_ratio: float,
        anti_smoothing_oscillation_floor: float,
        anti_smoothing_decay: float,
        anti_smoothing_target_amplitude_ratio: float,
        model: Any = None,
        dmd_prediction_diagnostics: dict[str, Any] | None = None,
        direct_fit_diagnostics: dict[str, Any] | None = None,
        direct_prediction_diagnostics: dict[str, Any] | None = None,
        weights: np.ndarray | None = None,
) -> dict[str, Any]:
    info = {
        'method': method_name,
        'head_runtime': 'fdmd' if uses_dmd(method_name) else 'direct_okhs',
        'q': resolved_q,
        'q_policy': q_policy,
        'forecast_horizon': forecast_horizon,
        'window_policy': window_policy,
        'trajectory_sampling_policy': trajectory_sampling_policy,
        'trajectory_rank_policy': trajectory_rank_policy,
        'trajectory_rank_value': trajectory_rank_value,
        'trajectory_representation_policy': trajectory_representation_policy,
        'latent_trajectory_stride_policy': latent_trajectory_stride_policy,
        'latent_trajectory_stride': latent_trajectory_stride,
        'resolved_window_size': resolved_window_size,
        'window_diagnostics': window_diagnostics,
        'trajectory_preprocessing': trajectory_preprocessing,
        'projection_metadata': projection_metadata,
        'mode_selection_policy': mode_selection_policy,
        'mode_energy_threshold': mode_energy_threshold,
        'prediction_mode_selection_policy': prediction_mode_selection_policy,
        'max_prediction_modes': max_prediction_modes,
        'min_prediction_modes': min_prediction_modes,
        'boundary_alignment_policy': boundary_alignment_policy,
        'boundary_alignment_decay': boundary_alignment_decay,
        'prediction_stability_threshold': prediction_stability_threshold,
        'anti_smoothing_policy': anti_smoothing_policy,
        'anti_smoothing_tail_window': anti_smoothing_tail_window,
        'anti_smoothing_amplitude_ratio': anti_smoothing_amplitude_ratio,
        'anti_smoothing_monotone_ratio': anti_smoothing_monotone_ratio,
        'anti_smoothing_oscillation_floor': anti_smoothing_oscillation_floor,
        'anti_smoothing_decay': anti_smoothing_decay,
        'anti_smoothing_target_amplitude_ratio': anti_smoothing_target_amplitude_ratio,
    }
    if model is not None and hasattr(model, 'get_fit_diagnostics_summary'):
        info['fdmd_fit_diagnostics'] = model.get_fit_diagnostics_summary()
    if dmd_prediction_diagnostics is not None:
        info['fdmd_prediction_diagnostics'] = dmd_prediction_diagnostics
    if direct_fit_diagnostics is not None:
        info['direct_fit_diagnostics'] = direct_fit_diagnostics
    if direct_prediction_diagnostics is not None:
        info['direct_prediction_diagnostics'] = direct_prediction_diagnostics
    if weights is not None:
        info['weights_norm'] = float(np.linalg.norm(weights))
    info['stage_diagnostics'] = build_okhs_stage_diagnostics(info)
    return info


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
) -> dict[str, Any]:
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
) -> tuple[np.ndarray, dict[str, Any]]:
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
        'envelope_ratio_after': diagnostics['envelope_ratio_before'],
        'collapse_still_detected_after': bool(diagnostics['collapse_detected']),
        'collapse_resolved': not bool(diagnostics['collapse_detected']),
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
    diagnostics['envelope_ratio_after'] = float(
        diagnostics['forecast_amplitude_after'] / max(float(diagnostics['train_tail_amplitude']), 1e-8)
    )
    diagnostics['collapse_still_detected_after'] = bool(
        float(diagnostics['train_tail_oscillation_score']) >= float(diagnostics['oscillation_floor'])
        and float(diagnostics['forecast_monotone_ratio_after']) >= float(diagnostics['monotone_ratio_threshold'])
        and float(diagnostics['envelope_ratio_after']) <= float(diagnostics['amplitude_ratio_threshold'])
    )
    diagnostics['collapse_resolved'] = not bool(diagnostics['collapse_still_detected_after'])
    return corrected, diagnostics


def postprocess_okhs_dmd_forecast(
        *,
        train_series: np.ndarray,
        forecast: np.ndarray,
        forecast_horizon: int,
        prediction_diagnostics: dict[str, Any] | None = None,
        anti_smoothing_policy: str = 'residual_bridge',
        anti_smoothing_tail_window: int | None = None,
        anti_smoothing_amplitude_ratio: float = 0.35,
        anti_smoothing_monotone_ratio: float = 0.9,
        anti_smoothing_oscillation_floor: float = 0.25,
        anti_smoothing_decay: float = 2.5,
        anti_smoothing_target_amplitude_ratio: float = 0.8,
) -> tuple[np.ndarray, dict[str, Any]]:
    corrected, anti_smoothing_diagnostics = apply_okhs_anti_smoothing(
        train_series=np.asarray(train_series, dtype=float).reshape(-1),
        forecast=np.asarray(forecast, dtype=float).reshape(-1),
        forecast_horizon=forecast_horizon,
        policy=anti_smoothing_policy,
        tail_window=anti_smoothing_tail_window,
        amplitude_ratio_threshold=anti_smoothing_amplitude_ratio,
        monotone_ratio_threshold=anti_smoothing_monotone_ratio,
        oscillation_floor=anti_smoothing_oscillation_floor,
        decay=anti_smoothing_decay,
        target_amplitude_ratio=anti_smoothing_target_amplitude_ratio,
    )
    merged_diagnostics = {
        **dict(prediction_diagnostics or {}),
        'anti_smoothing_diagnostics': anti_smoothing_diagnostics,
    }
    return corrected, merged_diagnostics
