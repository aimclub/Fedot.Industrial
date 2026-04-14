from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from fedot_ind.core.repository.forecasting_registry import canonical_forecasting_model_name


class ForecastingStageName(str, Enum):
    TRAJECTORY = 'trajectory_transform'
    DECOMPOSITION_RANK = 'decomposition_rank'
    FORECAST_HEAD = 'forecast_head'
    ENSEMBLE = 'ensemble'


@dataclass(frozen=True)
class StageTuningGroup:
    stage: str
    parameters: tuple[str, ...]
    recommended_tuner: str = 'SequentialTuner'
    depends_on: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForecastingStageTuningPlan:
    model_name: str
    canonical_model_name: str
    family: str
    groups: tuple[StageTuningGroup, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'model_name': self.model_name,
            'canonical_model_name': self.canonical_model_name,
            'family': self.family,
            'groups': [group.to_dict() for group in self.groups],
            **self.metadata,
        }


def _group(stage: ForecastingStageName, parameters: tuple[str, ...], *, depends_on: tuple[str, ...] = (),
           metadata: dict[str, Any] | None = None) -> StageTuningGroup:
    return StageTuningGroup(
        stage=stage.value,
        parameters=parameters,
        depends_on=depends_on,
        metadata=metadata or {},
    )


def build_forecasting_stage_tuning_plan(model_name: str,
                                        params: dict[str, Any] | None = None) -> ForecastingStageTuningPlan:
    canonical_name = canonical_forecasting_model_name(model_name)
    resolved_params = dict(params or {})

    if canonical_name in {'lagged_forecaster', 'lagged_ridge_forecaster'}:
        groups = (
            _group(ForecastingStageName.TRAJECTORY, ('window_size', 'window_size_percent', 'stride')),
            _group(
                ForecastingStageName.FORECAST_HEAD,
                ('alpha',),
                depends_on=(ForecastingStageName.TRAJECTORY.value,),
            ),
        )
        return ForecastingStageTuningPlan(
            model_name=model_name,
            canonical_model_name='lagged_ridge_forecaster',
            family='lagged_linear',
            groups=groups,
            metadata={'supports_simultaneous_tuning': True},
        )

    if canonical_name in {'ssa_forecaster', 'mssa_forecaster'}:
        groups = (
            _group(ForecastingStageName.TRAJECTORY, ('window_size',)),
            _group(
                ForecastingStageName.DECOMPOSITION_RANK,
                ('rank', 'explained_variance', 'lag_order', 'coupled', 'channel_independent'),
                depends_on=(ForecastingStageName.TRAJECTORY.value,),
            ),
        )
        return ForecastingStageTuningPlan(
            model_name=model_name,
            canonical_model_name=canonical_name,
            family='low_rank_linear',
            groups=groups,
            metadata={'supports_simultaneous_tuning': canonical_name == 'ssa_forecaster'},
        )

    if canonical_name == 'low_rank_lagged_ridge_forecaster':
        groups = (
            _group(ForecastingStageName.TRAJECTORY, ('window_size', 'window_size_percent', 'stride')),
            _group(
                ForecastingStageName.DECOMPOSITION_RANK,
                (
                'rank', 'explained_variance', 'decomposition_strategy', 'rank_truncation_policy', 'unfolding_strategy'),
                depends_on=(ForecastingStageName.TRAJECTORY.value,),
            ),
            _group(
                ForecastingStageName.FORECAST_HEAD,
                ('alpha',),
                depends_on=(
                    ForecastingStageName.TRAJECTORY.value,
                    ForecastingStageName.DECOMPOSITION_RANK.value,
                ),
            ),
        )
        return ForecastingStageTuningPlan(
            model_name=model_name,
            canonical_model_name=canonical_name,
            family='low_rank_linear',
            groups=groups,
            metadata={'supports_simultaneous_tuning': False},
        )

    if canonical_name == 'havok_forecaster':
        groups = (
            _group(ForecastingStageName.TRAJECTORY, ('window_size',)),
            _group(
                ForecastingStageName.DECOMPOSITION_RANK,
                ('rank',),
                depends_on=(ForecastingStageName.TRAJECTORY.value,),
            ),
            _group(
                ForecastingStageName.FORECAST_HEAD,
                ('forcing_threshold_scale', 'forcing_decay'),
                depends_on=(
                    ForecastingStageName.TRAJECTORY.value,
                    ForecastingStageName.DECOMPOSITION_RANK.value,
                ),
            ),
        )
        return ForecastingStageTuningPlan(
            model_name=model_name,
            canonical_model_name=canonical_name,
            family='operator_model',
            groups=groups,
            metadata={'supports_simultaneous_tuning': False},
        )

    if canonical_name in {'okhs', 'okhs_fdmd_forecaster'}:
        groups = (
            _group(
                ForecastingStageName.TRAJECTORY,
                (
                    'window_size',
                    'window_policy',
                    'trajectory_sampling_policy',
                    'latent_trajectory_stride_policy',
                    'latent_trajectory_stride',
                ),
            ),
            _group(
                ForecastingStageName.DECOMPOSITION_RANK,
                (
                    'trajectory_rank_policy',
                    'trajectory_rank_value',
                    'trajectory_representation_policy',
                ),
                depends_on=(ForecastingStageName.TRAJECTORY.value,),
            ),
            _group(
                ForecastingStageName.FORECAST_HEAD,
                (
                    'q',
                    'q_policy',
                    'n_modes',
                    'mode_selection_policy',
                    'mode_energy_threshold',
                    'prediction_mode_selection_policy',
                    'max_prediction_modes',
                    'min_prediction_modes',
                    'boundary_alignment_policy',
                    'boundary_alignment_decay',
                    'prediction_stability_threshold',
                    'anti_smoothing_policy',
                    'anti_smoothing_tail_window',
                    'anti_smoothing_amplitude_ratio',
                    'anti_smoothing_monotone_ratio',
                    'anti_smoothing_oscillation_floor',
                    'anti_smoothing_decay',
                    'anti_smoothing_target_amplitude_ratio',
                ),
                depends_on=(
                    ForecastingStageName.TRAJECTORY.value,
                    ForecastingStageName.DECOMPOSITION_RANK.value,
                ),
                metadata={'device': resolved_params.get('device', 'cpu')},
            ),
        )
        return ForecastingStageTuningPlan(
            model_name=model_name,
            canonical_model_name='okhs_fdmd_forecaster' if canonical_name != 'okhs' else 'okhs',
            family='operator_model',
            groups=groups,
            metadata={'supports_simultaneous_tuning': False},
        )

    if canonical_name == 'hybrid_ensemble_forecaster':
        groups = (
            _group(
                ForecastingStageName.TRAJECTORY,
                ('lagged_params', 'low_rank_params', 'complex_params'),
                metadata={
                    'branch_models': (
                        'lagged_ridge_forecaster',
                        'low_rank_lagged_ridge_forecaster',
                        resolved_params.get('complex_branch', 'okhs'),
                    ),
                },
            ),
            _group(
                ForecastingStageName.ENSEMBLE,
                ('complex_branch', 'calibration_horizon'),
                depends_on=(ForecastingStageName.TRAJECTORY.value,),
            ),
        )
        return ForecastingStageTuningPlan(
            model_name=model_name,
            canonical_model_name=canonical_name,
            family='hybrid_ensemble',
            groups=groups,
            metadata={'supports_simultaneous_tuning': False},
        )

    return ForecastingStageTuningPlan(
        model_name=model_name,
        canonical_model_name=canonical_name,
        family='unknown',
        groups=(),
        metadata={'supports_simultaneous_tuning': False},
    )
