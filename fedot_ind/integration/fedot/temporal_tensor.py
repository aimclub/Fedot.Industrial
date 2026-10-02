"""Effectful TensorData construction for validated temporal values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from fedot import create_data

from fedot_ind.integration.fedot.compatibility import (
    DataTypesEnum,
    OutputData,
    Task,
    TaskTypesEnum,
    TsForecastingParams,
    tensordata_to_input_data,
)
from fedot_ind.integration.fedot.contracts import (
    IntegrationContractError,
    IntegrationErrorCode,
)
from fedot_ind.integration.fedot.temporal_contracts import (
    DetectionPrediction,
    ForecastingPrediction,
    PreparedDetectionData,
    PreparedForecastingData,
)
from fedot_ind.integration.fedot.temporal_planning import intervals_to_labels


@dataclass(frozen=True)
class ForecastingTensorData:
    """Live TensorData paired with immutable forecasting coordinates."""

    tensor_data: Any
    prepared: PreparedForecastingData

    def to_input_data(self):
        return tensordata_to_input_data(self.tensor_data)


@dataclass(frozen=True)
class DetectionTensorData:
    """Live TensorData paired with immutable detection context."""

    tensor_data: Any
    prepared: PreparedDetectionData

    def to_input_data(self):
        input_data = tensordata_to_input_data(self.tensor_data)
        supplementary = getattr(input_data, "supplementary_data", None)
        if supplementary is not None:
            setattr(supplementary, "temporal_idx", np.array(
                self.prepared.time_idx, copy=True))
            if self.prepared.gap_mask is not None:
                setattr(supplementary, "gap_mask", np.array(
                    self.prepared.gap_mask, copy=True))
        return input_data


def create_forecasting_tensor_data(
        prepared: PreparedForecastingData,
        *,
        from_data: ForecastingTensorData | None = None,
) -> ForecastingTensorData:
    """Create TensorData while retaining time coordinates outside its object index."""
    schema = prepared.plan.schema
    if schema.series_count != 1 or schema.channel_count != 1:
        raise IntegrationContractError(
            IntegrationErrorCode.SCHEMA_MISMATCH,
            "The current forecasting runtime supports one univariate series.",
            context={
                "series_count": schema.series_count,
                "channel_count": schema.channel_count,
            },
        )
    task = Task(
        TaskTypesEnum.ts_forecasting,
        TsForecastingParams(forecast_length=prepared.plan.horizon),
    )
    source = None if from_data is None else from_data.tensor_data
    tensor_data = create_data(
        np.array(prepared.values[0], copy=True),
        task=task,
        data_type="ts",
        from_data=source,
        idx=np.array(prepared.sample_idx, copy=True),
        ts_orientation="wide",
        ts_forecast_horizon=prepared.plan.horizon,
        without_target=True,
    )
    return ForecastingTensorData(tensor_data=tensor_data, prepared=prepared)


def create_detection_tensor_data(
        prepared: PreparedDetectionData,
        *,
        from_data: DetectionTensorData | None = None,
) -> DetectionTensorData:
    """Create point-aligned TensorData for an anomaly detector."""
    source = None if from_data is None else from_data.tensor_data
    target = None
    if prepared.intervals:
        target = intervals_to_labels(
            prepared.intervals, prepared.plan.schema.history_length)
    tensor_data = create_data(
        np.array(prepared.values, copy=True),
        target=target,
        task="classification",
        data_type="tabular",
        from_data=source,
        idx=np.array(prepared.time_idx, copy=True),
        features_names=list(prepared.plan.schema.channel_names) or None,
        without_target=target is None,
    )
    return DetectionTensorData(tensor_data=tensor_data, prepared=prepared)


def forecasting_prediction_to_output_data(
        prediction: ForecastingPrediction,
        source: ForecastingTensorData,
) -> OutputData:
    """Restore a typed forecast as a FEDOT public ``OutputData`` value."""
    output = OutputData(
        idx=np.array(prediction.forecast_time_idx, copy=True),
        features=None,
        predict=np.array(prediction.values, copy=True),
        target=None,
        task=source.tensor_data.task,
        data_type=DataTypesEnum.table,
    )
    _attach_temporal_metadata(output, prediction.to_dict())
    return output


def detection_prediction_to_output_data(
        prediction: DetectionPrediction,
        source: DetectionTensorData,
) -> OutputData:
    """Restore point-aligned detector output without losing interval context."""
    output = OutputData(
        idx=np.array(prediction.time_idx, copy=True),
        features=None,
        predict=np.array(prediction.values, copy=True),
        target=None,
        task=source.tensor_data.task,
        data_type=DataTypesEnum.table,
    )
    _attach_temporal_metadata(output, prediction.to_dict())
    return output


def _attach_temporal_metadata(output: OutputData, metadata: dict[str, Any]) -> None:
    supplementary = getattr(output, "supplementary_data", None)
    if supplementary is not None:
        setattr(supplementary, "temporal_contract", metadata)


__all__ = [
    "DetectionTensorData",
    "ForecastingTensorData",
    "create_detection_tensor_data",
    "create_forecasting_tensor_data",
    "detection_prediction_to_output_data",
    "forecasting_prediction_to_output_data",
]
