from typing import Optional

from fedot.core.operations.evaluation.common_preprocessing import FedotPreprocessingStrategy
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.models.signal.RecurrenceExtractor import RecurrenceExtractor
from fedot_ind.core.models.signal.SignalExtractor import SignalExtractor
from fedot_ind.core.models.statistical.StatsExtractor import StatsExtractor
from fedot_ind.core.models.topological.TopologicalExtractor import TopologicalExtractor
from fedot_ind.core.operation.dummy.dummy_operation import DummyOperation
from fedot_ind.core.operation.transformation.basis.data_driven import DataDrivenBasisImplementation
from fedot_ind.core.operation.transformation.basis.fourier import FourierBasisImplementation
from fedot_ind.core.operation.transformation.basis.wavelet import WaveletBasisImplementation
from fedot_ind.core.repository.IndustrialOperationParameters import IndustrialOperationParameters


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
        'data_driven_basis': DataDrivenBasisImplementation,
        'wavelet_basis': WaveletBasisImplementation,
        'fourier_basis': FourierBasisImplementation,
        'topological_extractor': TopologicalExtractor,
        'quantile_extractor': StatsExtractor,
        'signal_extractor': SignalExtractor,
        'recurrence_extractor': RecurrenceExtractor,

        'cat_features': DummyOperation
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
