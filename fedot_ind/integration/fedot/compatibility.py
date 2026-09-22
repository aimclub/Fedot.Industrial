"""FEDOT data type imports shared by the explicit integration profiles."""

from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

try:
    from fedot.core.data.input_data.data import InputData, OutputData
except ModuleNotFoundError as error:
    if error.name not in {
            "fedot.core.data.input_data",
            "fedot.core.data.input_data.data",
    }:
        raise
    from fedot.core.data.data import InputData, OutputData

try:
    from fedot.core.data.multimodal.multi_modal import MultiModalData
except ModuleNotFoundError as error:
    if error.name not in {
            "fedot.core.data.multimodal",
            "fedot.core.data.multimodal.multi_modal",
    }:
        raise
    from fedot.core.data.multi_modal import MultiModalData


def tensordata_to_input_data(tensor_data):
    """Use the explicit FEDOT bridge while supporting the pre-1.0 module layout."""
    try:
        from fedot.core.data.bridges.tensor_to_input import tensordata_to_input_data as bridge
    except ModuleNotFoundError as error:
        if error.name not in {
                "fedot.core.data.bridges",
                "fedot.core.data.bridges.tensor_to_input",
        }:
            raise
        return InputData(
            idx=tensor_data.idx,
            features=tensor_data.features,
            target=tensor_data.target,
            task=tensor_data.task,
            data_type=tensor_data.data_type,
            features_names=getattr(tensor_data, "features_names", None),
        )
    return bridge(tensor_data)


__all__ = [
    "DataTypesEnum",
    "InputData",
    "MultiModalData",
    "OperationParameters",
    "OutputData",
    "Task",
    "TaskTypesEnum",
    "TsForecastingParams",
    "tensordata_to_input_data",
]
