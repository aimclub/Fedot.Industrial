"""TensorData FEDOT adapter; imported only in the ``fedot-tensor`` profile."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import LinearRegression

from fedot import create_data
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.extensions import (
    ArrayBackend,
    ExtensionManifest,
    ExternalModelSpec,
    ModelCapabilities,
    extension_scope,
)

from fedot_ind.integration.fedot.contracts import (
    DataProfile,
    DataStage,
    IntegrationTask,
    IntegrationContractError,
    IntegrationErrorCode,
    ModelExecutionPlan,
    MultimodalPreparationPlan,
    PredictionMode,
    PreparedData,
    PreparedMultimodalData,
)
from fedot_ind.integration.fedot.data import normalize_multimodal_input
from fedot_ind.integration.fedot.extensions.bootstrap import industrial_extension_scope
from fedot_ind.integration.fedot.planning import (
    build_multimodal_plan,
    validate_multimodal_prediction_plan,
)
from fedot_ind.integration.fedot.runtime import RegressionRuntime, SupervisedRuntime


_OPERATION_NAME = "industrial_linear_regression_smoke"


def _manifest() -> ExtensionManifest:
    capabilities = ModelCapabilities(
        tasks=(TaskTypesEnum.regression,),
        data_types=(DataTypesEnum.tabular,),
        tags=("linear", "industrial_integration"),
        backend=ArrayBackend.numpy,
    )
    specification = ExternalModelSpec(
        name=_OPERATION_NAME,
        factory=LinearRegression,
        capabilities=capabilities,
        description="Minimal Industrial/FEDOT TensorData integration model.",
    )
    return ExtensionManifest(
        name="fedot_ind_int01",
        version="1",
        models=(specification,),
    )


class TensorRegressionRuntime(RegressionRuntime):
    """Linear regression through TensorData and a scoped extension manifest."""

    profile = DataProfile.TENSOR

    def __init__(self) -> None:
        super().__init__()
        self._manifest = _manifest()
        self._pipeline = None
        self._train_data = None

    def _fit(self, data: PreparedData, target: np.ndarray) -> None:
        self._train_data = create_data(
            np.array(data.values, copy=True),
            target=np.array(target, copy=True),
            task="regression",
            data_type="tabular",
        )
        with extension_scope(self._manifest):
            self._pipeline = Pipeline(PipelineNode(_OPERATION_NAME))
            self._pipeline.fit(self._train_data)

    def _predict(self, data: PreparedData) -> np.ndarray:
        predict_data = create_data(
            np.array(data.values, copy=True),
            from_data=self._train_data,
        )
        with extension_scope(self._manifest):
            output = self._pipeline.predict(predict_data)
        values = output.predict
        return values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)

    def _close(self) -> None:
        self._pipeline = None
        self._train_data = None


class TensorSupervisedRuntime(SupervisedRuntime):
    """Execute a manifest-declared Industrial model through FEDOT TensorData."""

    def __init__(self, plan: ModelExecutionPlan) -> None:
        super().__init__(plan)
        self._pipeline = None
        self._train_data = None

    def _fit(self, data: PreparedData, target: np.ndarray) -> None:
        self._train_data = _create_tensor_data(data, target, self.plan.task)
        with industrial_extension_scope():
            self._pipeline = PipelineBuilder().add_node(
                self.plan.operation_name,
                params=self.plan.runtime_parameters(),
            ).build()
            self._pipeline.fit(self._train_data)

    def _predict(self, data: PreparedData, mode: PredictionMode) -> np.ndarray:
        predict_data = _create_tensor_data(
            data,
            None,
            self.plan.task,
            from_data=self._train_data,
        )
        with industrial_extension_scope():
            output = self._pipeline.predict(predict_data, output_mode=mode.value)
        values = output.predict
        return values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)

    def _close(self) -> None:
        self._pipeline = None
        self._train_data = None


@dataclass(frozen=True)
class TensorMultimodalBatch:
    """Live TensorData values paired with their pure preparation plan."""

    plan: MultimodalPreparationPlan
    modalities: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "modalities",
            MappingProxyType(dict(sorted(self.modalities.items()))),
        )


def create_multimodal_tensor_data(
        bundle,
        *,
        task: IntegrationTask | str,
        stage: DataStage | str,
        train_plan: MultimodalPreparationPlan | None = None,
        from_data: Mapping[str, Any] | None = None,
        idx: Any | None = None,
) -> TensorMultimodalBatch:
    """Convert an Industrial multimodal bundle into aligned FEDOT TensorData values."""
    payload = {
        modality.value: tensor.detach().cpu().numpy()
        for modality, tensor in bundle.modalities.items()
    }
    plan = build_multimodal_plan(payload, task=task, stage=stage)
    if train_plan is not None:
        validate_multimodal_prediction_plan(train_plan, plan)
    prepared = normalize_multimodal_input(payload, plan)
    if idx is not None:
        prepared = _replace_multimodal_index(prepared, idx)
    sources = dict(from_data or {})
    if sources and set(sources) != set(prepared.modalities):
        raise IntegrationContractError(
            IntegrationErrorCode.MODALITY_MISMATCH,
            "Fitted TensorData sources must match the prepared modalities.",
            context={
                "sources": sorted(sources),
                "modalities": sorted(prepared.modalities),
            },
        )
    target = None
    if bundle.target is not None:
        target = bundle.target.detach().cpu().numpy()
    tensor_data = {
        name: _create_tensor_data(
            modality,
            target,
            plan.task,
            from_data=sources.get(name),
        )
        for name, modality in prepared.modalities.items()
    }
    return TensorMultimodalBatch(plan=plan, modalities=tensor_data)


def _replace_multimodal_index(
        prepared: PreparedMultimodalData,
        idx: Any,
) -> PreparedMultimodalData:
    coordinates = np.asarray(idx)
    if coordinates.ndim != 1 or len(coordinates) != prepared.sample_count:
        raise IntegrationContractError(
            IntegrationErrorCode.INDEX_MISMATCH,
            "Explicit multimodal index must match the sample count.",
            context={
                "index_shape": list(coordinates.shape),
                "samples": prepared.sample_count,
            },
        )
    return PreparedMultimodalData(
        {
            name: PreparedData(
                values=modality.values,
                idx=coordinates,
                schema=modality.schema,
            )
            for name, modality in prepared.modalities.items()
        }
    )


def _create_tensor_data(
        data: PreparedData,
        target: np.ndarray | None,
        task: IntegrationTask,
        *,
        from_data=None,
):
    options = {
        "idx": np.array(data.idx, copy=True),
    }
    if data.schema.columns is not None:
        options["features_names"] = list(data.schema.columns)
    data_type = "tabular" if data.values.ndim <= 2 else "image"
    return create_data(
        np.array(data.values, copy=True),
        target=None if target is None else np.array(target, copy=True),
        task=task.value,
        data_type=data_type,
        from_data=from_data,
        **options,
    )
