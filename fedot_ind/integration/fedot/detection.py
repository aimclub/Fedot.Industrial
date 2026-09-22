"""TensorData-backed execution boundary for Industrial anomaly detectors."""

from __future__ import annotations

from typing import Any

import numpy as np
from fedot_ind.core.models.detection.runtime import (
    detect_events_from_score_series,
)
from fedot_ind.core.operation.interfaces.detection_runtime_strategy import (
    DETECTION_RUNTIME_MODELS,
    IndustrialDetectionModelRuntimeStrategy,
)
from fedot_ind.integration.fedot.contracts import (
    DataProfile,
    DataStage,
    IntegrationContractError,
    IntegrationErrorCode,
    RuntimeState,
)
from fedot_ind.integration.fedot.compatibility import OperationParameters
from fedot_ind.integration.fedot.extensions.catalog import load_industrial_extension_catalog
from fedot_ind.integration.fedot.temporal_contracts import (
    AnomalyInterval,
    DetectionExecutionPlan,
    DetectionMode,
    DetectionPrediction,
    PreparedDetectionData,
    TemporalOrientation,
    TemporalRuntimeSnapshot,
)
from fedot_ind.integration.fedot.temporal_planning import (
    labels_to_intervals,
    prepare_detection_data,
)
from fedot_ind.integration.fedot.temporal_tensor import (
    DetectionTensorData,
    create_detection_tensor_data,
    detection_prediction_to_output_data,
)


class TensorDetectionRuntime:
    """Execute one explicitly supported detector while preserving time context."""

    def __init__(self, plan: DetectionExecutionPlan) -> None:
        _validate_operation(plan.operation_name)
        self.plan = plan
        self._state = RuntimeState.CREATED
        self._strategy: IndustrialDetectionModelRuntimeStrategy | None = None
        self._operation: Any | None = None
        self._train_data: DetectionTensorData | None = None

    @property
    def snapshot(self) -> TemporalRuntimeSnapshot:
        return TemporalRuntimeSnapshot(
            profile=self.plan.profile,
            state=self._state,
            operation_name=self.plan.operation_name,
            plan=None if self._train_data is None else self._train_data.prepared.plan,
        )

    def fit(
            self,
            data,
            *,
            orientation: TemporalOrientation | str = TemporalOrientation.TIME_FIRST,
            time_idx=None,
            intervals=(),
            point_labels=None,
            gap_mask=None,
            channel_names=None,
    ) -> "TensorDetectionRuntime":
        self._require_state(RuntimeState.CREATED, "fit")
        prepared = data if isinstance(data, PreparedDetectionData) else prepare_detection_data(
            data,
            stage=DataStage.TRAIN,
            mode=self.plan.mode,
            orientation=orientation,
            time_idx=time_idx,
            intervals=intervals,
            point_labels=point_labels,
            gap_mask=gap_mask,
            channel_names=channel_names,
            causal=self.plan.causal,
        )
        if prepared.plan.stage is not DataStage.TRAIN:
            raise IntegrationContractError(
                IntegrationErrorCode.PLAN_MISMATCH,
                "Detection fit requires a training data plan.",
                context={"stage": prepared.plan.stage.value},
            )
        self._validate_data_plan(prepared)
        tensor_data = create_detection_tensor_data(prepared)
        parameters = self.plan.runtime_parameters()
        parameters["causal"] = self.plan.causal
        strategy = IndustrialDetectionModelRuntimeStrategy(
            self.plan.operation_name,
            params=OperationParameters(**parameters),
        )
        try:
            operation = strategy.fit(tensor_data.to_input_data())
        except Exception as error:
            raise _runtime_error(self.plan.operation_name,
                                 "fit", error) from error
        self._train_data = tensor_data
        self._strategy = strategy
        self._operation = operation
        self._state = RuntimeState.FITTED
        return self

    def predict(
            self,
            data,
            *,
            orientation: TemporalOrientation | str = TemporalOrientation.TIME_FIRST,
            time_idx=None,
            gap_mask=None,
            channel_names=None,
    ) -> DetectionPrediction:
        self._require_state(RuntimeState.FITTED, "predict")
        assert self._train_data is not None
        assert self._strategy is not None
        assert self._operation is not None
        prepared = data if isinstance(data, PreparedDetectionData) else prepare_detection_data(
            data,
            stage=DataStage.PREDICT,
            mode=self.plan.mode,
            orientation=orientation,
            time_idx=time_idx,
            gap_mask=gap_mask,
            channel_names=channel_names,
            causal=self.plan.causal,
        )
        if prepared.plan.stage is not DataStage.PREDICT:
            raise IntegrationContractError(
                IntegrationErrorCode.PLAN_MISMATCH,
                "Detection predict requires a prediction data plan.",
                context={"stage": prepared.plan.stage.value},
            )
        self._validate_data_plan(prepared)
        _validate_predict_schema(self._train_data.prepared, prepared)
        tensor_data = create_detection_tensor_data(
            prepared, from_data=self._train_data)
        input_data = tensor_data.to_input_data()
        try:
            values, intervals, metadata = self._predict_values(input_data)
        except Exception as error:
            raise _runtime_error(self.plan.operation_name,
                                 "predict", error) from error
        output_time_idx = np.asarray(
            getattr(self._operation, "last_prepared_time_idx_", prepared.time_idx)
        )
        return DetectionPrediction(
            values=values,
            time_idx=output_time_idx,
            mode=self.plan.mode,
            intervals=intervals,
            metadata={
                "operation_name": self.plan.operation_name,
                "input_time_index_preserved": bool(np.array_equal(output_time_idx, prepared.time_idx)),
                **metadata,
            },
        )

    def close(self) -> None:
        self._operation = None
        self._strategy = None
        self._train_data = None
        self._state = RuntimeState.CLOSED

    def predict_output_data(self, data, **kwargs):
        """Return point predictions through FEDOT's public output container."""
        prediction = self.predict(data, **kwargs)
        assert self._train_data is not None
        return detection_prediction_to_output_data(prediction, self._train_data)

    def _predict_values(self, input_data) -> tuple[np.ndarray, tuple[AnomalyInterval, ...], dict[str, Any]]:
        if hasattr(self._operation, "score_series_on_values"):
            score_series = self._operation.score_series_on_values(input_data)
            labels = np.asarray(score_series.labels, dtype=int)
            scores = np.asarray(score_series.scores, dtype=float)
            events = detect_events_from_score_series(
                score_series,
                min_event_length=int(
                    getattr(self._operation, "min_event_length", 1)
                ),
            )
            intervals = tuple(
                AnomalyInterval(start=event.start_index,
                                stop=event.end_index + 1, label=event.label)
                for event in events
            )
            if self.plan.mode is DetectionMode.LABELS:
                values = labels
            elif self.plan.mode is DetectionMode.SCORES:
                values = scores
            else:
                scale = float(getattr(
                    self._operation,
                    "score_reference_scale_",
                    max(abs(float(score_series.threshold)), 1.0),
                ))
                anomaly = 1.0 / (1.0 + np.exp(
                    -(scores - float(score_series.threshold)) / scale
                ))
                values = np.column_stack((1.0 - anomaly, anomaly))
            return values, intervals, {
                "threshold": float(score_series.threshold),
                "calibration_strategy": score_series.calibration_strategy,
            }

        output = self._strategy.predict(
            self._operation,
            input_data,
            output_mode=self.plan.mode.value,
        )
        raw = getattr(output, "predict", output)
        values = raw.detach().cpu().numpy() if hasattr(
            raw, "detach") else np.asarray(raw)
        values = np.asarray(values)
        intervals = ()
        if self.plan.mode is DetectionMode.LABELS:
            intervals = labels_to_intervals(values.reshape(-1))
        return values, intervals, {}

    def _require_state(self, expected: RuntimeState, phase: str) -> None:
        if self._state is not expected:
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_STATE,
                f"Detection runtime cannot {phase} in its current state.",
                context={"state": self._state.value,
                         "expected": expected.value},
            )

    def _validate_data_plan(self, prepared: PreparedDetectionData) -> None:
        mismatches = {}
        if prepared.plan.mode is not self.plan.mode:
            mismatches["mode"] = {
                "execution": self.plan.mode.value,
                "data": prepared.plan.mode.value,
            }
        if prepared.plan.causal is not self.plan.causal:
            mismatches["causal"] = {
                "execution": self.plan.causal,
                "data": prepared.plan.causal,
            }
        if mismatches:
            raise IntegrationContractError(
                IntegrationErrorCode.PLAN_MISMATCH,
                "Detection data plan must match the execution plan.",
                context={"mismatches": mismatches},
            )


def create_detection_runtime(
        *,
        profile: DataProfile | str,
        operation_name: str,
        mode: DetectionMode | str = DetectionMode.LABELS,
        causal: bool = True,
        parameters: dict[str, Any] | None = None,
) -> TensorDetectionRuntime:
    """Build a validated anomaly-detection runtime without a legacy fallback."""
    try:
        normalized_profile = profile if isinstance(
            profile, DataProfile) else DataProfile(profile)
    except (TypeError, ValueError) as error:
        raise IntegrationContractError(
            IntegrationErrorCode.UNKNOWN_PROFILE,
            "Detection runtime profile is unsupported.",
            context={"profile": profile},
            cause=error,
        ) from error
    try:
        normalized_mode = mode if isinstance(
            mode, DetectionMode) else DetectionMode(mode)
    except (TypeError, ValueError) as error:
        raise IntegrationContractError(
            IntegrationErrorCode.UNSUPPORTED_OUTPUT_MODE,
            "Detection output mode is unsupported.",
            context={"mode": mode},
            cause=error,
        ) from error
    _validate_causal_parameters(bool(causal), parameters or {})
    return TensorDetectionRuntime(DetectionExecutionPlan(
        profile=normalized_profile,
        operation_name=operation_name,
        mode=normalized_mode,
        causal=causal,
        parameters=parameters or {},
    ))


def _validate_causal_parameters(causal: bool, parameters: dict[str, Any]) -> None:
    if "causal" in parameters and bool(parameters["causal"]) is not causal:
        raise IntegrationContractError(
            IntegrationErrorCode.PLAN_MISMATCH,
            "The detector causal parameter must match the execution plan.",
            context={
                "execution": causal,
                "parameter": bool(parameters["causal"]),
            },
        )
    if not causal:
        return
    unsupported = {}
    gap_policy = str(parameters.get("gap_policy", "mark_only")).lower()
    if gap_policy == "interpolate_linear":
        unsupported["gap_policy"] = gap_policy
    transfer = str(parameters.get(
        "transfer_strategy",
        "domain_invariant_scaling",
    )).lower()
    if transfer in {"feature_alignment", "coral"}:
        unsupported["transfer_strategy"] = transfer
    if unsupported:
        raise IntegrationContractError(
            IntegrationErrorCode.UNSUPPORTED_OPERATION,
            "Causal detection does not allow transformations that use future values.",
            context={"parameters": unsupported},
        )


def _validate_operation(operation_name: str) -> None:
    declaration = load_industrial_extension_catalog().operation(operation_name)
    if (
            declaration is None
            or "anomaly_detection" not in declaration.problems
            or operation_name not in DETECTION_RUNTIME_MODELS
    ):
        raise IntegrationContractError(
            IntegrationErrorCode.UNSUPPORTED_OPERATION,
            "Detection operation is not available through the TensorData runtime.",
            context={"operation": operation_name},
        )


def _validate_predict_schema(train: PreparedDetectionData, predict: PreparedDetectionData) -> None:
    train_schema = train.plan.schema
    predict_schema = predict.plan.schema
    if (
            train_schema.channel_count != predict_schema.channel_count
            or train_schema.channel_names != predict_schema.channel_names
    ):
        raise IntegrationContractError(
            IntegrationErrorCode.SCHEMA_MISMATCH,
            "Detection prediction channels must match the fitted signal.",
            context={
                "train_channels": list(train_schema.channel_names),
                "predict_channels": list(predict_schema.channel_names),
            },
        )


def _runtime_error(operation: str, phase: str, error: Exception) -> IntegrationContractError:
    if isinstance(error, IntegrationContractError):
        return error
    return IntegrationContractError(
        IntegrationErrorCode.RUNTIME_FAILURE,
        "Detection runtime execution failed.",
        context={"operation": operation, "phase": phase,
                 "cause_type": type(error).__name__},
        cause=error,
    )


__all__ = ["TensorDetectionRuntime", "create_detection_runtime"]
