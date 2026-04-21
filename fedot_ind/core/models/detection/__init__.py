from fedot_ind.core.models.detection.runtime import (
    AnomalyScoreSeries,
    DetectionBoundaryAdapter,
    DetectionEvent,
    DetectionSeriesEvaluation,
    DetectionSplitKind,
    DetectionSplitSpec,
    DetectionWindowBatch,
    RegimeSegment,
    RiskFeatureFrame,
    TransferAlignmentReport,
)
from fedot_ind.core.models.detection.stage_tuning import (
    DetectionStageName,
    DetectionStageTuningPlan,
    StageTuningGroup,
    build_detection_stage_tuning_plan,
)

__all__ = [
    'AnomalyScoreSeries',
    'DetectionBoundaryAdapter',
    'DetectionEvent',
    'DetectionSeriesEvaluation',
    'DetectionSplitKind',
    'DetectionSplitSpec',
    'DetectionStageName',
    'DetectionStageTuningPlan',
    'DetectionWindowBatch',
    'RegimeSegment',
    'RiskFeatureFrame',
    'StageTuningGroup',
    'TransferAlignmentReport',
    'build_detection_stage_tuning_plan',
]

try:  # pragma: no cover - optional FEDOT runtime dependency
    from fedot_ind.core.models.detection.modern_detectors import (
        BaseRuntimeAnomalyDetector,
        ConvAutoencoderDetector,
        FeatureIsolationForestDetector,
        FeatureOneClassDetector,
        TCNAutoencoderDetector,
        build_detection_input_data,
    )

    __all__.extend(
        [
            'BaseRuntimeAnomalyDetector',
            'ConvAutoencoderDetector',
            'FeatureIsolationForestDetector',
            'FeatureOneClassDetector',
            'TCNAutoencoderDetector',
            'build_detection_input_data',
        ]
    )
except Exception:  # pragma: no cover
    pass
