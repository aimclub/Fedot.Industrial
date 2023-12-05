from typing import Optional, Union

from fedot.core.data.data import InputData
from fedot.core.operations.evaluation.common_preprocessing import FedotPreprocessingStrategy
from fedot.core.operations.operation import Operation
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.models.nn.network_impl.mini_rocket import MiniRocketExtractor
from fedot_ind.core.models.recurrence.reccurence_extractor import RecurrenceExtractor
from fedot_ind.core.models.signal.signal_extractor import SignalExtractor
from fedot_ind.core.models.quantile.quantile_extractor import QuantileExtractor
from fedot_ind.core.models.topological.topological_extractor import TopologicalExtractor
from fedot_ind.core.operation.dummy.dummy_operation import DummyOperation
from fedot_ind.core.operation.filtration.feature_filtration import FeatureFilter

from fedot_ind.core.operation.transformation.basis.fourier import FourierBasisImplementation
from fedot_ind.core.operation.transformation.basis.wavelet import WaveletBasisImplementation
from fedot_ind.core.operation.transformation.basis.eigen_basis import EigenBasisImplementation

from fedot_ind.core.repository.IndustrialOperationParameters import IndustrialOperationParameters


class IndustrialBaseStrategy(Operation):
    """
    Args:
        operation_type: ``str`` of the operation defined in operation or data operation repositories

            .. details:: possible operations:

                - ``data_driven_basic``-> EigenBasisImplementation,
                - ``topological_features``-> TopologicalExtractor,


        params: hyperparameters to fit the operation with

    """

    __operations_by_types = {
        'eigen_basis': EigenBasisImplementation,
        'wavelet_basis': WaveletBasisImplementation,
        'fourier_basis': FourierBasisImplementation,
        'topological_extractor': TopologicalExtractor,
        'quantile_extractor': QuantileExtractor,
        'signal_extractor': SignalExtractor,
        'recurrence_extractor': RecurrenceExtractor,
        'minirocket_extractor': MiniRocketExtractor,
        'cat_features': DummyOperation,
        'dimension_reduction': FeatureFilter
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        params = IndustrialOperationParameters().from_params(operation_type, params) if params \
            else IndustrialOperationParameters().from_operation_type(operation_type)
        super().__init__(operation_type, params)

    def predict(self, fitted_operation, data: InputData, params: Optional[Union[OperationParameters, dict]] = None,
                output_mode: str = 'labels'):
        """This method is used for defining and running of the evaluation strategy
        to predict with the data provided

        Args:
            fitted_operation: trained operation object
            data: data used for prediction
            params: hyperparameters for operation
            output_mode: string with information about output of operation,
            for example, is the operation predict probabilities or class labels
        """
        return self._predict(fitted_operation, data, params, output_mode, is_fit_stage=False)

    def predict_for_fit(self, fitted_operation, data: InputData, params: Optional[OperationParameters] = None,
                        output_mode: str = 'labels'):
        """This method is used for defining and running of the evaluation strategy
        to predict with the data provided during fit stage

        Args:
            fitted_operation: trained operation object
            data: data used for prediction
            params: hyperparameters for operation
            output_mode: string with information about output of operation,
                for example, is the operation predict probabilities or class labels
        """
        return self._predict(fitted_operation, data, params, output_mode, is_fit_stage=True)