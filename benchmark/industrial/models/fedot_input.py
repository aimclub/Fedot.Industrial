"""Lazy conversion at the supervised FEDOT model boundary."""
import numpy as np


def fedot_input_data(features: np.ndarray, target: np.ndarray, *, task_type: str):
    from fedot.core.data.input_data.data import InputData
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum

    task = Task(TaskTypesEnum.classification if task_type == 'classification' else TaskTypesEnum.regression)
    return InputData(
        idx=np.arange(features.shape[0]),
        features=features,
        target=target,
        task=task,
        data_type=DataTypesEnum.table,
    )
