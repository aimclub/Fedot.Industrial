from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class ForecastingVerbosityLevel(str, Enum):
    """Supported verbosity presets for forecasting benchmark artifacts."""

    COMPACT = 'compact'
    STANDARD = 'standard'
    DEBUG = 'debug'


@dataclass(frozen=True)
class ForecastingVerbosityPolicy:
    """Policy that prunes heavy benchmark metadata before serialization."""

    level: str = ForecastingVerbosityLevel.STANDARD.value
    include_stage_tuning_report: bool = True
    include_stage_tuning_comparison: bool = True
    include_tuned_metadata: bool = True
    include_tuned_forecast: bool = True
    include_fold_details: bool = True
    include_stage_candidate_evaluations: bool = True
    include_progress_policy: bool = True
    include_runner_context: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize verbosity settings for run metadata."""
        return asdict(self)

    def prune_stage_tuning_report(self, report: dict[str, Any] | None) -> dict[str, Any] | None:
        """Remove heavy stage tuning report fields according to policy."""
        if not self.include_stage_tuning_report or not isinstance(report, dict):
            return None
        payload = dict(report)
        sequential = dict(payload.get('sequential_result') or {})
        if not self.include_stage_candidate_evaluations:
            stage_history = []
            for stage in sequential.get('stage_history', []):
                item = dict(stage)
                item.pop('evaluations', None)
                stage_history.append(item)
            sequential['stage_history'] = stage_history
        if not self.include_progress_policy:
            metadata = dict(sequential.get('metadata') or {})
            metadata.pop('progress_policy', None)
            if metadata:
                sequential['metadata'] = metadata
            else:
                sequential.pop('metadata', None)
        payload['sequential_result'] = sequential

        for key in ('baseline_evaluation', 'best_evaluation'):
            evaluation = dict(payload.get(key) or {})
            if not self.include_fold_details:
                split_metadata = dict(evaluation.get('split_metadata') or {})
                split_metadata.pop('folds', None)
                if split_metadata:
                    evaluation['split_metadata'] = split_metadata
                else:
                    evaluation.pop('split_metadata', None)
            if not self.include_progress_policy:
                metadata = dict(evaluation.get('metadata') or {})
                metadata.pop('progress_policy', None)
                if metadata:
                    evaluation['metadata'] = metadata
                else:
                    evaluation.pop('metadata', None)
            payload[key] = evaluation

        if not self.include_progress_policy:
            payload.pop('progress_policy', None)
        return payload

    def prune_stage_tuning_runtime(self, runtime_payload: dict[str, Any] | None) -> dict[str, Any] | None:
        """Remove heavy runtime fields according to policy."""
        if runtime_payload is None:
            return None
        payload = dict(runtime_payload)
        if not self.include_progress_policy:
            payload.pop('progress_policy', None)
        return payload

    def prune_stage_tuning_comparison(self, comparison: dict[str, Any] | None) -> dict[str, Any] | None:
        """Remove heavy post-fit tuning comparison fields according to policy."""
        if not self.include_stage_tuning_comparison or not isinstance(comparison, dict):
            return None
        payload = dict(comparison)
        if not self.include_tuned_metadata:
            payload.pop('tuned_metadata', None)
        if not self.include_tuned_forecast:
            payload.pop('tuned_forecast', None)
        return payload


def resolve_forecasting_verbosity_policy(
        level: str | ForecastingVerbosityLevel | None = None,
        *,
        options: dict[str, Any] | None = None,
) -> ForecastingVerbosityPolicy:
    """Resolve a verbosity preset plus overrides into a concrete policy."""
    resolved_level = ForecastingVerbosityLevel(str(level or ForecastingVerbosityLevel.STANDARD.value).lower())
    defaults = {
        ForecastingVerbosityLevel.COMPACT: dict(
            include_stage_tuning_report=True,
            include_stage_tuning_comparison=True,
            include_tuned_metadata=False,
            include_tuned_forecast=False,
            include_fold_details=False,
            include_stage_candidate_evaluations=False,
            include_progress_policy=False,
            include_runner_context=False,
        ),
        ForecastingVerbosityLevel.STANDARD: dict(
            include_stage_tuning_report=True,
            include_stage_tuning_comparison=True,
            include_tuned_metadata=True,
            include_tuned_forecast=True,
            include_fold_details=True,
            include_stage_candidate_evaluations=True,
            include_progress_policy=True,
            include_runner_context=False,
        ),
        ForecastingVerbosityLevel.DEBUG: dict(
            include_stage_tuning_report=True,
            include_stage_tuning_comparison=True,
            include_tuned_metadata=True,
            include_tuned_forecast=True,
            include_fold_details=True,
            include_stage_candidate_evaluations=True,
            include_progress_policy=True,
            include_runner_context=True,
        ),
    }[resolved_level]
    merged = {**defaults, **dict(options or {})}
    return ForecastingVerbosityPolicy(
        level=resolved_level.value,
        include_stage_tuning_report=bool(merged['include_stage_tuning_report']),
        include_stage_tuning_comparison=bool(merged['include_stage_tuning_comparison']),
        include_tuned_metadata=bool(merged['include_tuned_metadata']),
        include_tuned_forecast=bool(merged['include_tuned_forecast']),
        include_fold_details=bool(merged['include_fold_details']),
        include_stage_candidate_evaluations=bool(merged['include_stage_candidate_evaluations']),
        include_progress_policy=bool(merged['include_progress_policy']),
        include_runner_context=bool(merged['include_runner_context']),
    )

class DetectionVerbosityLevel(str, Enum):
    """Supported verbosity presets for anomaly-detection benchmark artifacts."""

    COMPACT = 'compact'
    STANDARD = 'standard'
    DEBUG = 'debug'


@dataclass(frozen=True)
class DetectionVerbosityPolicy:
    """Policy that prunes heavy detection benchmark metadata before serialization.

    The policy controls what remains in run metadata when stage-tuning payloads
    are attached. This allows choosing between compact outputs for CI and rich
    outputs for debugging/research.
    """

    level: str = DetectionVerbosityLevel.STANDARD.value
    include_stage_tuning_report: bool = True
    include_stage_tuning_comparison: bool = True
    include_tuned_metadata: bool = True
    include_tuned_labels: bool = True
    include_fold_details: bool = True
    include_stage_candidate_evaluations: bool = True
    include_progress_policy: bool = True
    include_runner_context: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize policy fields into a metadata-friendly dictionary."""
        return asdict(self)

    def prune_stage_tuning_report(self, report: dict[str, Any] | None) -> dict[str, Any] | None:
        """Prune heavy fields from the stage-tuning report payload.

        Parameters
        ----------
        report:
            Full report returned by detection stage tuning runtime.
        """
        if not self.include_stage_tuning_report or not isinstance(report, dict):
            return None
        payload = dict(report)
        sequential = dict(payload.get('sequential_result') or {})
        if not self.include_stage_candidate_evaluations:
            stage_history = []
            for stage in sequential.get('stage_history', []):
                item = dict(stage)
                item.pop('evaluations', None)
                stage_history.append(item)
            sequential['stage_history'] = stage_history
        if not self.include_progress_policy:
            metadata = dict(sequential.get('metadata') or {})
            metadata.pop('progress_policy', None)
            if metadata:
                sequential['metadata'] = metadata
            else:
                sequential.pop('metadata', None)
        payload['sequential_result'] = sequential

        for key in ('baseline_evaluation', 'best_evaluation'):
            evaluation = dict(payload.get(key) or {})
            if not self.include_fold_details:
                split_metadata = dict(evaluation.get('split_metadata') or {})
                split_metadata.pop('folds', None)
                if split_metadata:
                    evaluation['split_metadata'] = split_metadata
                else:
                    evaluation.pop('split_metadata', None)
            if not self.include_progress_policy:
                metadata = dict(evaluation.get('metadata') or {})
                metadata.pop('progress_policy', None)
                if metadata:
                    evaluation['metadata'] = metadata
                else:
                    evaluation.pop('metadata', None)
            payload[key] = evaluation

        if not self.include_progress_policy:
            payload.pop('progress_policy', None)
        return payload

    def prune_stage_tuning_runtime(self, runtime_payload: dict[str, Any] | None) -> dict[str, Any] | None:
        """Prune runtime-level tuning metadata based on verbosity switches."""
        if runtime_payload is None:
            return None
        payload = dict(runtime_payload)
        if not self.include_progress_policy:
            payload.pop('progress_policy', None)
        return payload

    def prune_stage_tuning_comparison(self, comparison: dict[str, Any] | None) -> dict[str, Any] | None:
        """Prune baseline-vs-tuned comparison payload for detection runs."""
        if not self.include_stage_tuning_comparison or not isinstance(comparison, dict):
            return None
        payload = dict(comparison)
        if not self.include_tuned_metadata:
            payload.pop('tuned_metadata', None)
        if not self.include_tuned_labels:
            payload.pop('tuned_labels', None)
        return payload


def resolve_detection_verbosity_policy(
        level: str | DetectionVerbosityLevel | None = None,
        *,
        options: dict[str, Any] | None = None,
) -> DetectionVerbosityPolicy:
    """Resolve detection verbosity preset plus option overrides.

    Parameters
    ----------
    level:
        Preset name (`compact`, `standard`, `debug`) or enum value.
    options:
        Optional explicit overrides merged on top of preset defaults.
    """
    resolved_level = DetectionVerbosityLevel(str(level or DetectionVerbosityLevel.STANDARD.value).lower())
    defaults = {
        DetectionVerbosityLevel.COMPACT: dict(
            include_stage_tuning_report=True,
            include_stage_tuning_comparison=True,
            include_tuned_metadata=False,
            include_tuned_labels=False,
            include_fold_details=False,
            include_stage_candidate_evaluations=False,
            include_progress_policy=False,
            include_runner_context=False,
        ),
        DetectionVerbosityLevel.STANDARD: dict(
            include_stage_tuning_report=True,
            include_stage_tuning_comparison=True,
            include_tuned_metadata=True,
            include_tuned_labels=True,
            include_fold_details=True,
            include_stage_candidate_evaluations=True,
            include_progress_policy=True,
            include_runner_context=False,
        ),
        DetectionVerbosityLevel.DEBUG: dict(
            include_stage_tuning_report=True,
            include_stage_tuning_comparison=True,
            include_tuned_metadata=True,
            include_tuned_labels=True,
            include_fold_details=True,
            include_stage_candidate_evaluations=True,
            include_progress_policy=True,
            include_runner_context=True,
        ),
    }[resolved_level]
    merged = {**defaults, **dict(options or {})}
    return DetectionVerbosityPolicy(
        level=resolved_level.value,
        include_stage_tuning_report=bool(merged['include_stage_tuning_report']),
        include_stage_tuning_comparison=bool(merged['include_stage_tuning_comparison']),
        include_tuned_metadata=bool(merged['include_tuned_metadata']),
        include_tuned_labels=bool(merged['include_tuned_labels']),
        include_fold_details=bool(merged['include_fold_details']),
        include_stage_candidate_evaluations=bool(merged['include_stage_candidate_evaluations']),
        include_progress_policy=bool(merged['include_progress_policy']),
        include_runner_context=bool(merged['include_runner_context']),
    )


