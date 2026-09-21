"""TensorData FEDOT adapter; imported only in the ``fedot-tensor`` profile."""

import numpy as np
from sklearn.linear_model import LinearRegression

from fedot import create_data
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.extensions import (
    ArrayBackend,
    ExtensionManifest,
    ExternalModelSpec,
    ModelCapabilities,
    extension_scope,
)

from fedot_ind.integration.fedot.contracts import DataProfile, PreparedData
from fedot_ind.integration.fedot.runtime import RegressionRuntime


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
