from pathlib import Path
from typing import Optional

from fedot.core.operations.evaluation.common_preprocessing import FedotPreprocessingStrategy
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.operation_types_repository import OperationTypesRepository

from core.architecture.utils.utils import PROJECT_PATH
from core.models.signal.RecurrenceExtractor import RecurrenceExtractor
from core.models.signal.SignalExtractor import SignalExtractor
from core.models.statistical.QuantileRunner import StatsExtractor
from core.models.topological.TopologicalRunner import TopologicalExtractor
from core.operation.transformation.basis.data_driven import DataDrivenBasisImplementation
from core.repository.IndustrialOperationParameters import IndustrialOperationParameters


class IndustrialPreprocessingStrategy(FedotPreprocessingStrategy):
    """
    Args:
        operation_type: ``str`` of the operation defined in operation or data operation repositories

            .. details:: possible operations:

                - ``data_driven_basic``-> DataDrivenBasisImplementation,
                - ``topological_features``-> TopologicalExtractor,


        params: hyperparameters to fit the operation with

    """

    __operations_by_types = {
        'data_driven_basic': DataDrivenBasisImplementation,
        'topological_extractor': TopologicalExtractor,
        'quantile_extractor': StatsExtractor,
        'signal_extractor': SignalExtractor,
        'recurrence_extractor': RecurrenceExtractor
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        params = IndustrialOperationParameters().from_params(operation_type, params) if params \
            else IndustrialOperationParameters().from_operation_type(operation_type)
        super().__init__(operation_type, params)

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain custom preprocessing strategy for {operation_type}')