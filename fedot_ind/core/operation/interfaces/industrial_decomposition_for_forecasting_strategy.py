from typing import Optional

from fedot.core.operations.evaluation.time_series import FedotTsForecastingStrategy
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.models.ts_forecasting.ssa_forecaster import SSAForecasterImplementation
from fedot_ind.core.repository.IndustrialOperationParameters import IndustrialOperationParameters


class IndustrialDecompositionForForecastingStrategy(FedotTsForecastingStrategy):
    """
    Args:
        operation_type: ``str`` of the operation defined in operation or data operation repositories

            .. details:: possible operations:

                - ``data_driven_basic``-> EigenBasisImplementation,
                - ``topological_features``-> TopologicalExtractor,


        params: hyperparameters to fit the operation with

    """

    __operations_by_types = {
        'ssa_forecaster': SSAForecasterImplementation,
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
