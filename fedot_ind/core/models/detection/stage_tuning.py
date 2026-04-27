from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from fedot_ind.core.repository.detection_registry import canonical_detection_model_name, detection_family_for


class DetectionStageName(str, Enum):
    DATA_QUALITY = 'data_quality'
    REGIME_SEGMENTATION = 'regime_segmentation'
    REPRESENTATION = 'representation'
    ANOMALY_SCORING = 'anomaly_scoring'
    CALIBRATION = 'calibration'
    EVENT_AGGREGATION = 'event_aggregation'
    TRANSFER_ALIGNMENT = 'transfer_alignment'
    INTERPRETATION = 'interpretation'


FALLBACK_DETECTION_STAGE_PARAMETERS: dict[str, dict[str, tuple[str, ...]]] = {
    'feature_iforest_detector': {
        DetectionStageName.REPRESENTATION.value: ('window_size', 'stride', 'representation_mode'),
        DetectionStageName.ANOMALY_SCORING.value: ('n_estimators', 'contamination', 'random_state'),
        DetectionStageName.CALIBRATION.value: ('calibration_strategy', 'threshold_quantile'),
        DetectionStageName.EVENT_AGGREGATION.value: ('min_event_length',),
    },
    'feature_oneclass_detector': {
        DetectionStageName.REPRESENTATION.value: ('window_size', 'stride', 'representation_mode'),
        DetectionStageName.ANOMALY_SCORING.value: ('nu', 'kernel', 'gamma'),
        DetectionStageName.CALIBRATION.value: ('calibration_strategy', 'threshold_quantile'),
        DetectionStageName.EVENT_AGGREGATION.value: ('min_event_length',),
    },
    'conv_autoencoder_detector': {
        DetectionStageName.REPRESENTATION.value: ('window_size', 'stride'),
        DetectionStageName.ANOMALY_SCORING.value: ('epochs', 'batch_size', 'learning_rate', 'latent_dim'),
        DetectionStageName.CALIBRATION.value: ('calibration_strategy', 'threshold_quantile'),
        DetectionStageName.EVENT_AGGREGATION.value: ('min_event_length',),
        DetectionStageName.TRANSFER_ALIGNMENT.value: ('transfer_strategy',),
    },
    'tcn_autoencoder_detector': {
        DetectionStageName.REPRESENTATION.value: ('window_size', 'stride'),
        DetectionStageName.ANOMALY_SCORING.value: (
            'epochs',
            'batch_size',
            'learning_rate',
            'latent_dim',
            'kernel_size',
            'num_filters',
            'num_levels',
        ),
        DetectionStageName.CALIBRATION.value: ('calibration_strategy', 'threshold_quantile'),
        DetectionStageName.EVENT_AGGREGATION.value: ('min_event_length',),
        DetectionStageName.TRANSFER_ALIGNMENT.value: ('transfer_strategy',),
    },
}


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
class DetectionStageTuningPlan:
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


def _group(
        stage: DetectionStageName,
        parameters: tuple[str, ...],
        *,
        depends_on: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
) -> StageTuningGroup:
    return StageTuningGroup(
        stage=stage.value,
        parameters=parameters,
        depends_on=depends_on,
        metadata=metadata or {},
    )


def build_detection_stage_tuning_plan(
        model_name: str,
        params: dict[str, Any] | None = None,
) -> DetectionStageTuningPlan:
    del params
    canonical_name = canonical_detection_model_name(model_name)
    family = detection_family_for(canonical_name)
    groups_payload = FALLBACK_DETECTION_STAGE_PARAMETERS.get(canonical_name, {})
    groups = []
    for stage_name, parameters in groups_payload.items():
        depends_on: tuple[str, ...] = ()
        if stage_name in {
            DetectionStageName.ANOMALY_SCORING.value,
            DetectionStageName.CALIBRATION.value,
            DetectionStageName.EVENT_AGGREGATION.value,
        }:
            depends_on = (DetectionStageName.REPRESENTATION.value,)
        if stage_name == DetectionStageName.EVENT_AGGREGATION.value:
            depends_on = (
                DetectionStageName.REPRESENTATION.value,
                DetectionStageName.CALIBRATION.value,
            )
        groups.append(
            _group(
                DetectionStageName(stage_name),
                parameters,
                depends_on=depends_on,
            )
        )
    return DetectionStageTuningPlan(
        model_name=model_name,
        canonical_model_name=canonical_name,
        family=family,
        groups=tuple(groups),
        metadata={'supports_stage_tuning': bool(groups)},
    )
