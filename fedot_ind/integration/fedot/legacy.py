"""Legacy FEDOT adapter; imported only in the ``fedot-legacy`` profile."""

import numpy as np

from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

from fedot_ind.integration.fedot.contracts import DataProfile, PreparedData
from fedot_ind.integration.fedot.runtime import RegressionRuntime


class LegacyRegressionRuntime(RegressionRuntime):
    """Linear regression through the stable InputData/Pipeline contract."""

    profile = DataProfile.LEGACY

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = None
        self._task = Task(TaskTypesEnum.regression)

    def _fit(self, data: PreparedData, target: np.ndarray) -> None:
        self._pipeline = PipelineBuilder().add_node("linear").build()
        self._pipeline.fit(self._input(data, target))

    def _predict(self, data: PreparedData) -> np.ndarray:
        output = self._pipeline.predict(self._input(data, None))
        return np.asarray(output.predict)

    def _input(self, data: PreparedData, target: np.ndarray | None) -> InputData:
        return InputData(
            idx=np.array(data.idx, copy=True),
            features=np.array(data.values, copy=True),
            target=None if target is None else np.array(target, copy=True),
            task=self._task,
            data_type=DataTypesEnum.table,
        )

    def _close(self) -> None:
        self._pipeline = None
