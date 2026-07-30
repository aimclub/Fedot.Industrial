from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

import numpy as np

from fedot_ind.core.kernel_learning.contracts import FeatureInput, KernelBundle, KernelTaskType
from fedot_ind.core.operation.transformation.torch_backend.io import (
    normalize_time_series_tensor,
    resolve_torch_device,
    to_numpy,
    to_torch,
)

DEFAULT_GENERATOR_NAMES = (
    "quantile_extractor",
    "wavelet_extractor",
    "fourier_extractor",
    "eigen_extractor",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OperationSpec:
    name: str
    module_path: str
    class_name: str
    params: dict[str, Any] = field(default_factory=dict)
    use_torch: bool = False


def normalize_feature_matrix(values: Any) -> np.ndarray:
    array = _to_numpy(values)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return np.nan_to_num(array.astype(float, copy=False), nan=0.0, posinf=0.0, neginf=0.0)


def to_fedot_input_data(
        X: Any,
        y: Any | None = None,
        *,
        task_type: str = "classification",
        use_torch: bool = False,
        torch_device: Any = "auto",
):
    from fedot.core.data.data import InputData
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum

    features = normalize_time_series_tensor(X)
    if use_torch:
        features = _to_torch(features, device=torch_device)
    task_map = {
        "classification": Task(TaskTypesEnum.classification),
        "regression": Task(TaskTypesEnum.regression),
        "ts_forecasting": Task(TaskTypesEnum.ts_forecasting),
    }
    if task_type not in task_map:
        raise ValueError(f"Unsupported supervised task_type: {task_type}")
    target = None if y is None else np.asarray(y).reshape(-1, 1)
    return InputData(
        idx=np.arange(features.shape[0]),
        features=features,
        target=target,
        task=task_map[task_type],
        data_type=DataTypesEnum.image,
    )


def features_to_fedot_input_data(features: Any, template: Any, *, use_torch: bool, torch_device: Any = "auto"):
    from fedot.core.data.data import InputData
    from fedot.core.repository.dataset_types import DataTypesEnum

    prepared = _to_torch(
        features, device=torch_device) if use_torch else _to_numpy(features)
    data_type = DataTypesEnum.image if len(
        prepared.shape) >= 3 else DataTypesEnum.table
    return InputData(
        idx=getattr(template, "idx", np.arange(prepared.shape[0])),
        features=prepared,
        target=getattr(template, "target", None),
        task=getattr(template, "task", None),
        data_type=data_type,
        supplementary_data=getattr(template, "supplementary_data", None),
    )


def unwrap_operation_output(value: Any) -> Any:
    prediction = getattr(value, "predict", None)
    if prediction is not None and not callable(prediction):
        return prediction
    return value


def operation_params(params: dict[str, Any]):
    try:
        from fedot.core.operations.operation_parameters import OperationParameters

        return OperationParameters(**params)
    except Exception:
        logger.exception(
            "Falling back to raw operation params for kernel feature generator operation.")
        return params


def load_operation_class(spec: OperationSpec):
    module = import_module(spec.module_path)
    return getattr(module, spec.class_name)


def call_transform(operation: Any, input_data: Any):
    transform = operation.transform
    parameters = inspect.signature(transform).parameters
    if "use_cache" in parameters:
        return transform(input_data, use_cache=False)
    return transform(input_data)


class KernelFeatureGeneratorMixin:
    """Default feature-generator kernel contract via features -> kernel builder."""

    def kernel(
            self,
            X_left: FeatureInput,
            X_right: FeatureInput | None = None,
            *,
            task_type: KernelTaskType | str | None = None,
    ) -> KernelBundle:
        from fedot_ind.core.kernel_learning.kernels import KernelMatrixBuilder

        resolved_task_type = task_type or getattr(
            self, "task_type_", None) or "classification"
        train_bundle = self.fit_transform(
            X_left, None, task_type=resolved_task_type)
        builder = KernelMatrixBuilder()
        kernel_bundle = builder.fit_transform(
            train_bundle.features,
            name=train_bundle.name,
            train_features=train_bundle.features,
        )
        if X_right is None:
            return kernel_bundle
        test_bundle = self.transform(X_right)
        return builder.build_test_bundle(test_bundle.features, kernel_bundle)


_features_to_fedot_input_data = features_to_fedot_input_data
_unwrap_operation_output = unwrap_operation_output
_to_numpy = to_numpy
_to_torch = to_torch
_operation_params = operation_params
_load_operation_class = load_operation_class
_call_transform = call_transform


__all__ = [
    "DEFAULT_GENERATOR_NAMES",
    "KernelFeatureGeneratorMixin",
    "OperationSpec",
    "call_transform",
    "features_to_fedot_input_data",
    "load_operation_class",
    "normalize_feature_matrix",
    "normalize_time_series_tensor",
    "operation_params",
    "resolve_torch_device",
    "to_fedot_input_data",
    "to_numpy",
    "to_torch",
    "unwrap_operation_output",
]
