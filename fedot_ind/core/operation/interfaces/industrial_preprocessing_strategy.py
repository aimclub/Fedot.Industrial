import warnings
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.common_preprocessing import FedotPreprocessingStrategy
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy, \
    convert_to_multivariate_model, is_multi_output_task
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import \
    *
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.utilities.random import ImplementationRandomStateHandler
from fedot_ind.core.architecture.settings.constanst_repository import INDUSTRIAL_CLF_PREPROC_MODEL, SKLEARN_CLF_MODELS, \
    FEDOT_PREPROC_MODEL, INDUSTRIAL_PREPROC_MODEL
from fedot_ind.core.repository.IndustrialOperationParameters import IndustrialOperationParameters


class MultiDimPreprocessingStrategy(FedotPreprocessingStrategy):
    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def _find_operation_by_impl(self, impl):
        for operation, operation_impl in self._operations_by_types.items():
            if operation_impl == impl:
                return operation

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))

    def _sklearn_compatible_prediction(self, trained_operation, features):
        is_multi_output_target = isinstance(trained_operation.classes_, list)
        # Check if target is multilabel (has 2 or more columns)
        if is_multi_output_target:
            n_classes = len(trained_operation.classes_[0])
        else:
            n_classes = len(trained_operation.classes_)
        if self.output_mode == 'labels':
            prediction = trained_operation.predict(features)
        elif self.output_mode in ['probs', 'full_probs', 'default']:
            prediction = trained_operation.predict_proba(features)
            if n_classes < 2:
                raise ValueError('Data set contain only 1 target class. Please reformat your data.')
            elif n_classes == 2 and self.output_mode != 'full_probs':
                if is_multi_output_target:
                    prediction = np.stack([pred[:, 1] for pred in prediction]).T
                else:
                    prediction = prediction[:, 1]
        else:
            raise ValueError(f'Output model {self.output_mode} is not supported')

        return prediction

    def _convert_input_data(self, train_data, mode: str = 'feature_extraction'):
        if len(train_data.features.shape) > 2:
            if mode == 'model_fitting':
                new_shape = train_data.features.shape[0], train_data.features.shape[1] * train_data.features.shape[2]
                features = train_data.features.reshape(new_shape)
                input_data = InputData(idx=train_data.idx,
                                       features=features,
                                       target=train_data.target,
                                       task=train_data.task,
                                       data_type=train_data.data_type,
                                       supplementary_data=train_data.supplementary_data)
            else:
                input_data = [InputData(idx=train_data.idx,
                                        features=features,
                                        target=train_data.target,
                                        task=train_data.task,
                                        data_type=train_data.data_type,
                                        supplementary_data=train_data.supplementary_data) for features in
                              train_data.features.swapaxes(1, 0)]
        else:
            input_data = train_data
        return input_data

    def _predict_for_ndim(self, predict_data, trained_operation):
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

    def _custom_fit(self, train_data):
        operation_implementation = self.operation_impl(self.params_for_fit)
        fit_method = operation_implementation.fit
        with ImplementationRandomStateHandler(implementation=operation_implementation):
            if type(train_data) is list:
                fitted_operation = list(map(fit_method, train_data))
                operation_implementation = fitted_operation
            else:
                operation_implementation = fit_method(train_data)
        return operation_implementation

    def fit_one_sample(self, train_data: InputData):
        """This method is used for operation training with the data provided

        Args:
            train_data: data used for operation training

        Returns:
            trained Sklearn operation
        """

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        operation_implementation = self.operation_impl(**self.params_for_fit.to_dict())

        # If model doesn't support multi-output and current task is ts_forecasting
        current_task = train_data.task.task_type
        models_repo = OperationTypesRepository()
        non_multi_models = models_repo.suitable_operation(task_type=current_task,
                                                          tags=['non_multi'])
        is_model_not_support_multi = self.operation_type in non_multi_models

        # Multi-output task or not
        is_multi_target = is_multi_output_task(train_data)
        with ImplementationRandomStateHandler(implementation=operation_implementation):
            if is_model_not_support_multi and is_multi_target:
                # Manually wrap the regressor into multi-output model
                operation_implementation = convert_to_multivariate_model(operation_implementation,
                                                                         train_data)
            else:
                operation_implementation.fit(train_data.features, train_data.target)
        return operation_implementation

    def fit(self, train_data: InputData, mode: str = 'model_fitting'):
        """This method is used for operation training with the data provided

        Args:
            train_data: data used for operation training

        Returns:
            trained Sklearn operation
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        train_data = self._convert_input_data(train_data, mode=mode)

        if mode == 'model_fitting':
            return self.fit_one_sample(train_data)
        elif mode == 'feature_extraction':
            operation_implementation = self.operation_impl(self.params_for_fit)
            with ImplementationRandomStateHandler(implementation=operation_implementation):
                if type(train_data) is list:
                    fitted_operation = [operation_implementation for i in range(len(train_data))]
                    operation_implementation = fitted_operation
            return operation_implementation
        else:
            return self._custom_fit(train_data)



    def predict_for_fit(self, trained_operation, predict_data: InputData, mode: str = 'model_fitting') -> OutputData:
        """
        Transform method for preprocessing task for fit stage

        Args:
            trained_operation: model object
            predict_data: data used for prediction
        Returns:
            OutputData:
        """

        if mode == 'model_fitting':
            if len(predict_data.features.shape) > 2:
                predict_data = self._convert_input_data(predict_data, mode='model_fitting')
            prediction = self._sklearn_compatible_prediction(trained_operation=trained_operation,
                                                             features=predict_data.features)
        else:
            if type(trained_operation) is list:
                prediction = self._predict_for_ndim(predict_data, trained_operation)
            else:
                prediction = trained_operation.transform(predict_data)
        converted = self._convert_to_output(prediction, predict_data, output_data_type=predict_data.data_type)
        return converted

    def predict(self, trained_operation, predict_data: InputData, mode: str = 'model_fitting') -> OutputData:
        """Method to predict the target data for predict stage.

        Args:
            trained_operation: trained operation object
            predict_data: data to predict

        Returns:
            passed data with new predicted target
        """
        predict_data = self._convert_input_data(predict_data)
        if mode == 'model_fitting':
            prediction = self.predict(trained_operation, predict_data)
        else:
            if type(trained_operation) is list:
                prediction = self._predict_for_ndim(predict_data, trained_operation)
            else:
                prediction = trained_operation.transform(predict_data)
        converted = self._convert_to_output(prediction, predict_data, output_data_type=predict_data.data_type)
        return converted


class IndustrialPreprocessingStrategy(FedotPreprocessingStrategy):
    __operations_by_types = INDUSTRIAL_PREPROC_MODEL

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
    _operations_by_types = FEDOT_PREPROC_MODEL

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        self.multi_dim_dispatcher = MultiDimPreprocessingStrategy(self.operation_impl)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        return self.multi_dim_dispatcher.fit(train_data, mode='custom_fit')

    def predict(self, trained_operation, predict_data: InputData):
        return self.multi_dim_dispatcher.predict(trained_operation, predict_data, mode='feature_extraction')

    def predict_for_fit(self, trained_operation, predict_data: InputData) -> OutputData:
        return self.multi_dim_dispatcher.predict_for_fit(trained_operation, predict_data, mode='feature_extraction')


class IndustrialSkLearnEvaluationStrategy(EvaluationStrategy):
    _operations_by_types = SKLEARN_CLF_MODELS

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        self.multi_dim_dispatcher = MultiDimPreprocessingStrategy(self.operation_impl)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        return self.multi_dim_dispatcher.fit(train_data, mode='model_fitting')

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        return self.multi_dim_dispatcher.predict(trained_operation, predict_data, mode='model_fitting')

    def predict_for_fit(self, trained_operation, predict_data: InputData) -> OutputData:
        return self.multi_dim_dispatcher.predict_for_fit(trained_operation, predict_data, mode='model_fitting')


class IndustrialSkLearnClassificationStrategy(IndustrialSkLearnEvaluationStrategy):
    """ Strategy for applying classification algorithms from Sklearn library """

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        self.multi_dim_dispatcher = MultiDimPreprocessingStrategy(self.operation_impl)
        super().__init__(operation_type, params)

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        return self.multi_dim_dispatcher.predict(trained_operation, predict_data, mode='model_fitting')

    def fit(self, train_data: InputData):
        return self.multi_dim_dispatcher.fit(train_data, mode='model_fitting')

    def predict_for_fit(self, trained_operation, predict_data: InputData) -> OutputData:
        return self.multi_dim_dispatcher.predict_for_fit(trained_operation, predict_data, mode='model_fitting')


class IndustrialClassificationPreprocessingStrategy(IndustrialCustomPreprocessingStrategy):
    """ Strategy for applying custom algorithms from FEDOT to preprocess data
    for classification task
    """

    _operations_by_types = INDUSTRIAL_CLF_PREPROC_MODEL

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)
        self.multi_dim_dispatcher = MultiDimPreprocessingStrategy(self.operation_impl)

    def fit(self, train_data: InputData):
        return self.multi_dim_dispatcher.fit(train_data, mode='feature_extraction')

    def predict_for_fit(self, trained_operation, predict_data: InputData) -> OutputData:
        return self.multi_dim_dispatcher.predict_for_fit(trained_operation, predict_data, mode='feature_extraction')
