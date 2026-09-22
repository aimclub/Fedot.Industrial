"""FEDOT data type imports shared by the explicit integration profiles."""

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


__all__ = ["InputData", "MultiModalData", "OutputData"]
