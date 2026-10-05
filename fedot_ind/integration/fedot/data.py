"""Pure numpy/pandas normalization for the FEDOT integration boundary."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fedot_ind.integration.fedot.contracts import (
    DataPreparationPlan,
    IntegrationContractError,
    IntegrationErrorCode,
    MultimodalPreparationPlan,
    PreparedData,
    PreparedMultimodalData,
)
from fedot_ind.integration.fedot.planning import (
    SupportedData,
    SupportedMultimodalData,
    build_data_plan,
    build_multimodal_plan,
)


def normalize_input_data(data: SupportedData, plan: DataPreparationPlan) -> PreparedData:
    """Apply a validated plan while preserving sample count, index, and source data."""
    observed_plan = build_data_plan(
        data,
        profile=plan.profile,
        task=plan.task,
        stage=plan.stage,
        axes=plan.axes,
    )
    if observed_plan.source_rank != plan.source_rank or observed_plan.feature_schema != plan.feature_schema:
        raise IntegrationContractError(
            IntegrationErrorCode.SCHEMA_MISMATCH,
            "Input data no longer matches the preparation plan.",
            context={
                "planned_rank": plan.source_rank,
                "observed_rank": observed_plan.source_rank,
                "planned_schema": plan.feature_schema.to_dict(),
                "observed_schema": observed_plan.feature_schema.to_dict(),
            },
        )
    values = data.to_numpy(copy=True) if isinstance(data, (pd.Series, pd.DataFrame)) else np.array(data, copy=True)
    idx = data.index.to_numpy(copy=True) if isinstance(data, (pd.Series, pd.DataFrame)) else np.arange(
        values.shape[plan.axes.sample],
    )
    normalized = _canonical_values(values, plan)
    return PreparedData(values=normalized, idx=idx, schema=plan.feature_schema)


def normalize_multimodal_input(
        data: SupportedMultimodalData,
        plan: MultimodalPreparationPlan,
) -> PreparedMultimodalData:
    """Apply modality plans and require exact index alignment."""
    observed = build_multimodal_plan(
        data,
        task=plan.task,
        stage=plan.stage,
    )
    if tuple(observed.modalities) != tuple(plan.modalities):
        raise IntegrationContractError(
            IntegrationErrorCode.MODALITY_MISMATCH,
            "Input modalities no longer match the preparation plan.",
            context={
                "planned": list(plan.modalities),
                "observed": list(observed.modalities),
            },
        )
    prepared = {
        name: normalize_input_data(data[name], modality_plan)
        for name, modality_plan in plan.modalities.items()
    }
    return PreparedMultimodalData(prepared)


def _canonical_values(values: np.ndarray, plan: DataPreparationPlan) -> np.ndarray:
    axes = plan.axes
    if values.ndim == 1:
        return values.reshape(values.shape[0], 1)
    if values.ndim == 2:
        return np.moveaxis(values, (axes.sample, axes.feature), (0, 1))
    return np.moveaxis(values, (axes.sample, axes.channel, axes.feature), (0, 1, 2))
