"""Stable FEDOT imports and data bridges owned by the integration boundary."""

from typing import Any

import numpy as np

from fedot import create_data
from fedot.core.data.bridges.input_to_tensor import input_data_to_tensordata
from fedot.core.data.bridges.tensor_to_input import tensordata_to_input_data
from fedot.core.data.common.enums import StateEnum
from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.data.multimodal.multi_modal import MultiModalData
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


def ensure_fedot_tensor_data(
        data: Any,
        *,
        fit_stage: bool,
        reference_data: TensorData | None = None,
) -> Any:
    """Convert legacy ``InputData`` at a current FEDOT runtime boundary.

    Industrial implementations still use ``InputData`` internally.  The current
    FEDOT API, pipelines and tuners use ``TensorData``.  Unknown values are kept
    unchanged so lightweight service collaborators and third-party strategies can
    retain their own data contract.
    """
    if isinstance(data, TensorData) or not isinstance(data, InputData):
        return data

    if reference_data is not None:
        options = {
            "features_names": data.features_names,
        }
        if np.asarray(data.features).ndim != 1:
            options["idx"] = data.idx
        return create_data(
            data.features,
            target=data.target,
            from_data=reference_data,
            **options,
        )

    is_legacy_forecast_series = (
        data.data_type == DataTypesEnum.ts
        and np.asarray(data.features).ndim == 1
        and data.idx is not None
        and len(data.idx) != 1
    )
    if is_legacy_forecast_series:
        if not fit_stage:
            raise ValueError(
                "Forecast prediction conversion requires reference_data from the fitted FEDOT model."
            )
        horizon = int(data.task.task_params.forecast_length)
        return create_data(
            data.features,
            task=data.task,
            data_type=data.data_type,
            ts_forecast_horizon=horizon,
        )

    state = StateEnum.FIT if fit_stage else StateEnum.PREDICT
    return input_data_to_tensordata(data, backend_name="cpu", state=state)


def as_numpy(value: Any) -> np.ndarray:
    """Return a CPU NumPy view for a FEDOT or Industrial runtime value."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


__all__ = [
    "DataTypesEnum",
    "InputData",
    "MultiModalData",
    "OperationParameters",
    "OutputData",
    "Task",
    "TaskTypesEnum",
    "TensorData",
    "TsForecastingParams",
    "as_numpy",
    "ensure_fedot_tensor_data",
    "tensordata_to_input_data",
]
