"""Pure construction and validation of FEDOT data preparation plans."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, TypeVar

import numpy as np
import pandas as pd

from fedot_ind.integration.fedot.contracts import (
    AxisLayout,
    DataPreparationPlan,
    DataProfile,
    DataStage,
    FeatureSchema,
    IntegrationContractError,
    IntegrationErrorCode,
    IntegrationTask,
    MultimodalPreparationPlan,
)


EnumType = TypeVar("EnumType", bound=Enum)
SupportedData = np.ndarray | pd.Series | pd.DataFrame
SupportedMultimodalData = Mapping[str, SupportedData]


def build_data_plan(
        data: SupportedData,
        *,
        profile: DataProfile | str,
        task: IntegrationTask | str,
        stage: DataStage | str,
        axes: AxisLayout | None = None,
) -> DataPreparationPlan:
    """Build a deterministic plan without changing the supplied data."""
    normalized_profile = _parse_enum(profile, DataProfile, IntegrationErrorCode.UNKNOWN_PROFILE, "profile")
    normalized_task = _parse_enum(task, IntegrationTask, IntegrationErrorCode.UNKNOWN_TASK, "task")
    normalized_stage = _parse_enum(stage, DataStage, IntegrationErrorCode.UNKNOWN_STAGE, "stage")
    source_shape = _source_shape(data)
    source_rank = len(source_shape)
    normalized_axes = axes or _default_axes(normalized_profile, source_rank)
    _validate_axes(normalized_profile, normalized_axes, source_rank)
    _validate_pandas_axes(data, normalized_axes)
    schema = _feature_schema(data, normalized_profile, normalized_axes)
    return DataPreparationPlan(
        profile=normalized_profile,
        task=normalized_task,
        stage=normalized_stage,
        source_rank=source_rank,
        axes=normalized_axes,
        feature_schema=schema,
    )


def validate_prediction_plan(train_plan: DataPreparationPlan, predict_plan: DataPreparationPlan) -> None:
    """Require prediction data to follow the fitted plan and feature schema."""
    if train_plan.stage is not DataStage.TRAIN or predict_plan.stage is not DataStage.PREDICT:
        raise IntegrationContractError(
            IntegrationErrorCode.PLAN_MISMATCH,
            "Prediction validation requires train and predict plans in that order.",
            context={"train_stage": train_plan.stage.value, "predict_stage": predict_plan.stage.value},
        )
    comparable_fields = ("profile", "task")
    mismatches = {
        field: {
            "train": getattr(train_plan, field).value,
            "predict": getattr(predict_plan, field).value,
        }
        for field in comparable_fields
        if getattr(train_plan, field) is not getattr(predict_plan, field)
    }
    if mismatches:
        raise IntegrationContractError(
            IntegrationErrorCode.PLAN_MISMATCH,
            "Prediction profile and task must match the training plan.",
            context={"mismatches": mismatches},
        )
    if train_plan.feature_schema != predict_plan.feature_schema:
        raise IntegrationContractError(
            IntegrationErrorCode.SCHEMA_MISMATCH,
            "Prediction features must match the training feature schema.",
            context={
                "train_schema": train_plan.feature_schema.to_dict(),
                "predict_schema": predict_plan.feature_schema.to_dict(),
            },
        )


def build_multimodal_plan(
        data: SupportedMultimodalData,
        *,
        task: IntegrationTask | str,
        stage: DataStage | str,
) -> MultimodalPreparationPlan:
    """Build deterministic TensorData plans for named modalities."""
    if not isinstance(data, Mapping) or not data:
        raise IntegrationContractError(
            IntegrationErrorCode.INVALID_DATA,
            "Multimodal input must be a non-empty mapping.",
            context={"type": type(data).__name__},
        )
    normalized_task = _parse_enum(
        task, IntegrationTask, IntegrationErrorCode.UNKNOWN_TASK, "task"
    )
    normalized_stage = _parse_enum(
        stage, DataStage, IntegrationErrorCode.UNKNOWN_STAGE, "stage"
    )
    plans: dict[str, DataPreparationPlan] = {}
    sample_counts: dict[str, int] = {}
    for raw_name, values in data.items():
        if (
                not isinstance(raw_name, str)
                or not raw_name.strip()
                or raw_name != raw_name.strip()
        ):
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_DATA,
                "Modality names must be non-empty trimmed strings.",
                context={"modality": raw_name},
            )
        name = raw_name
        if name in plans:
            raise IntegrationContractError(
                IntegrationErrorCode.MODALITY_MISMATCH,
                "Normalized modality names must be unique.",
                context={"modality": name},
            )
        plans[name] = build_data_plan(
            values,
            profile=DataProfile.TENSOR,
            task=normalized_task,
            stage=normalized_stage,
        )
        sample_counts[name] = _sample_count(values, plans[name].axes)
    if len(set(sample_counts.values())) != 1:
        raise IntegrationContractError(
            IntegrationErrorCode.LENGTH_MISMATCH,
            "All modalities must contain the same number of samples.",
            context={"sample_counts": sample_counts},
        )
    return MultimodalPreparationPlan(
        profile=DataProfile.TENSOR,
        task=normalized_task,
        stage=normalized_stage,
        modalities=plans,
    )


def validate_multimodal_prediction_plan(
        train_plan: MultimodalPreparationPlan,
        predict_plan: MultimodalPreparationPlan,
) -> None:
    """Require the same modality set and feature schemas at prediction time."""
    train_names = tuple(train_plan.modalities)
    predict_names = tuple(predict_plan.modalities)
    if train_names != predict_names:
        raise IntegrationContractError(
            IntegrationErrorCode.MODALITY_MISMATCH,
            "Prediction modalities must match the training modalities.",
            context={"train": list(train_names), "predict": list(predict_names)},
        )
    for name in train_names:
        validate_prediction_plan(
            train_plan.modalities[name],
            predict_plan.modalities[name],
        )


def _parse_enum(value: Any, enum_type: type[EnumType], code: IntegrationErrorCode, field: str) -> EnumType:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise IntegrationContractError(
            code,
            f"Unsupported {field} value.",
            context={"field": field, "value": value, "allowed": [item.value for item in enum_type]},
        ) from exc


def _source_shape(data: SupportedData) -> tuple[int, ...]:
    if not isinstance(data, (np.ndarray, pd.Series, pd.DataFrame)):
        raise IntegrationContractError(
            IntegrationErrorCode.INVALID_DATA,
            "Input data must be a numpy array, pandas Series, or pandas DataFrame.",
            context={"type": type(data).__name__},
        )
    shape = tuple(int(size) for size in data.shape)
    if not shape or any(size < 1 for size in shape):
        raise IntegrationContractError(
            IntegrationErrorCode.INVALID_DATA,
            "Input data must contain at least one sample and one value.",
            context={"shape": shape},
        )
    return shape


def _sample_count(data: SupportedData, axes: AxisLayout) -> int:
    return int(data.shape[axes.sample])


def _default_axes(profile: DataProfile, rank: int) -> AxisLayout:
    if profile is DataProfile.LEGACY:
        if rank == 1:
            return AxisLayout(sample=0)
        if rank == 2:
            return AxisLayout(sample=0, feature=1)
    elif rank == 1:
        return AxisLayout(sample=0)
    elif rank == 2:
        return AxisLayout(sample=0, feature=1)
    elif rank == 3:
        return AxisLayout(sample=0, channel=1, feature=2)
    raise IntegrationContractError(
        IntegrationErrorCode.INVALID_AXES,
        "Input rank is not supported by the selected profile.",
        context={"profile": profile.value, "rank": rank},
    )


def _validate_axes(profile: DataProfile, axes: AxisLayout, rank: int) -> None:
    expected_axes = {
        DataProfile.LEGACY: {1: ("sample",), 2: ("sample", "feature")},
        DataProfile.TENSOR: {
            1: ("sample",),
            2: ("sample", "feature"),
            3: ("sample", "channel", "feature"),
        },
    }
    required = expected_axes[profile].get(rank)
    if required is None:
        raise IntegrationContractError(
            IntegrationErrorCode.INVALID_AXES,
            "Input rank is not supported by the selected profile.",
            context={"profile": profile.value, "rank": rank},
        )
    supplied = {name: getattr(axes, name) for name in ("sample", "channel", "feature")}
    invalid_presence = [name for name, value in supplied.items() if (name in required) != (value is not None)]
    positions = [supplied[name] for name in required]
    out_of_range = [position for position in positions if position is None or not 0 <= position < rank]
    if invalid_presence or out_of_range or len(set(positions)) != len(positions):
        raise IntegrationContractError(
            IntegrationErrorCode.INVALID_AXES,
            "Axis layout does not match the selected profile and input rank.",
            context={
                "profile": profile.value,
                "rank": rank,
                "required": list(required),
                "axes": axes.to_dict(),
            },
        )


def _validate_pandas_axes(data: SupportedData, axes: AxisLayout) -> None:
    if isinstance(data, pd.Series):
        expected = AxisLayout(sample=0)
    elif isinstance(data, pd.DataFrame):
        expected = AxisLayout(sample=0, feature=1)
    else:
        return
    if axes != expected:
        raise IntegrationContractError(
            IntegrationErrorCode.INVALID_AXES,
            "Pandas rows are the sample axis and columns are the feature axis.",
            context={"expected": expected.to_dict(), "axes": axes.to_dict()},
        )


def _feature_schema(data: SupportedData, profile: DataProfile, axes: AxisLayout) -> FeatureSchema:
    shape = tuple(int(size) for size in data.shape)
    columns = _column_names(data)
    if len(shape) == 1:
        dimensions = (1,)
    elif len(shape) == 2:
        dimensions = (shape[axes.feature],)
    else:
        dimensions = (shape[axes.channel], shape[axes.feature])
    return FeatureSchema(dimensions=dimensions, columns=columns)


def _column_names(data: SupportedData) -> tuple[str, ...] | None:
    if isinstance(data, pd.DataFrame):
        columns = tuple(str(column) for column in data.columns)
        if len(set(columns)) != len(columns):
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_DATA,
                "DataFrame feature names must be unique.",
                context={"columns": list(columns)},
            )
        return columns
    if isinstance(data, pd.Series):
        return (str(data.name) if data.name is not None else "0",)
    return None
