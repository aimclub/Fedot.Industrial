import warnings
from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.common_preprocessing import FedotPreprocessingStrategy
from fedot.core.operations.evaluation.operation_implementations.data_operations.categorical_encoders import \
    OneHotEncodingImplementation, LabelEncodingImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import \
    *
from fedot.core.operations.evaluation.operation_implementations.data_operations.topological.topological_extractor import \
    TopologicalFeaturesImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.utilities.random import ImplementationRandomStateHandler

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


class IndustrialPreprocessingStrategy(FedotPreprocessingStrategy):
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

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain custom preprocessing strategy for {operation_type}')


class IndustrialCustomPreprocessingStrategy(FedotPreprocessingStrategy):
    """
    Args:
        operation_type: ``str`` of the operation defined in operation or data operation repositories

            .. details:: possible operations:

                - ``scaling``-> ScalingImplementation,
                - ``normalization``-> NormalizationImplementation,
                - ``simple_imputation``-> ImputationImplementation,
                - ``pca``-> PCAImplementation,
                - ``kernel_pca``-> KernelPCAImplementation,
                - ``poly_features``-> PolyFeaturesImplementation,
                - ``one_hot_encoding``-> OneHotEncodingImplementation,
                - ``label_encoding``-> LabelEncodingImplementation,
                - ``fast_ica``-> FastICAImplementation

        params: hyperparameters to fit the operation with

    """

    _operations_by_types = {
        'scaling': ScalingImplementation,
        'normalization': NormalizationImplementation,
        'simple_imputation': ImputationImplementation,
        'pca': PCAImplementation,
        'kernel_pca': KernelPCAImplementation,
        'poly_features': PolyFeaturesImplementation,
        'one_hot_encoding': OneHotEncodingImplementation,
        'label_encoding': LabelEncodingImplementation,
        'fast_ica': FastICAImplementation,
        'topological_features': TopologicalFeaturesImplementation
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        """This method is used for operation training with the data provided

        Args:
            train_data: data used for operation training

        Returns:
            trained Sklearn operation
        """

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        operation_implementation = self.operation_impl(self.params_for_fit)
        with ImplementationRandomStateHandler(implementation=operation_implementation):
            if len(train_data.features.shape) > 2:
                input_data = [InputData(idx=train_data.idx,
                                        features=features,
                                        target=train_data.target,
                                        task=train_data.task,
                                        data_type=train_data.data_type,
                                        supplementary_data=train_data.supplementary_data) for features in
                              train_data.features.swapaxes(1, 0)]
                fitted_operation = list(map(operation_implementation.fit, input_data))
                operation_implementation = fitted_operation
            else:
                operation_implementation.fit(train_data)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """Transform method for preprocessing task

        Args:
            trained_operation: model object
            predict_data: data used for prediction

        Returns:
            prediction
        """
        if type(trained_operation) is list:
            prediction = self.__predict_for_ndim(predict_data, trained_operation)
        else:
            prediction = trained_operation.transform(predict_data)
        # Convert prediction to output (if it is required)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def __predict_for_ndim(self, predict_data, trained_operation):
        test_data = [InputData(idx=predict_data.idx,
                               features=features,
                               target=predict_data.target,
                               task=predict_data.task,
                               data_type=predict_data.data_type,
                               supplementary_data=predict_data.supplementary_data) for features in
                     predict_data.features.swapaxes(1, 0)]
        prediction = list(operation.transform(data.features) for operation, data in zip(trained_operation, test_data))
        prediction = np.stack(prediction).swapaxes(0, 1)
        return prediction

    def predict_for_fit(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        Transform method for preprocessing task for fit stage

        Args:
            trained_operation: model object
            predict_data: data used for prediction
        Returns:
            OutputData:
        """
        if type(trained_operation) is list:
            prediction = self.__predict_for_ndim(predict_data, trained_operation)
        else:
            prediction = trained_operation.transform(predict_data)
        converted = self._convert_to_output(prediction, predict_data)
        return converted
