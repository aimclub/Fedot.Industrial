import warnings
from copy import deepcopy

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy, \
    convert_to_multivariate_model, is_multi_output_task
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import \
    *
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operation_type_from_id
from fedot.utilities.random import ImplementationRandomStateHandler

from fedot_ind.core.architecture.preprocessing.data_convertor import NumpyConverter
from fedot_ind.core.repository.model_repository import INDUSTRIAL_CLF_PREPROC_MODEL, SKLEARN_CLF_MODELS, \
    FEDOT_PREPROC_MODEL, INDUSTRIAL_PREPROC_MODEL, SKLEARN_REG_MODELS
from fedot_ind.core.repository.IndustrialOperationParameters import IndustrialOperationParameters


class MultiDimPreprocessingStrategy(EvaluationStrategy):
    def __init__(self, operation_impl, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = operation_impl
        super().__init__(operation_type, params)
        self.output_mode = 'labels'

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))

    def __convert_input_data(self, input_data):
        if len(input_data.features.shape) > 2:
            converted_data = self._convert_input_data(input_data, mode='model_fitting')
        else:
            converted_data = input_data
        return converted_data

    def __operation_multidim_adapter(self, trained_operation, predict_data):
        if type(trained_operation) is list:
            prediction = self._predict_for_ndim(predict_data, trained_operation)
        else:
            try:
                prediction = trained_operation.transform(predict_data.features)
            except Exception:
                prediction = trained_operation.transform(predict_data)
        return prediction

    def __convert_dimensions(self, predict_data, prediction, output_mode: str = 'default'):

        multi_dim_features = len(predict_data.features.shape) > 2
        prob_prediction = len(prediction.shape) == 2
        sklearn_output_mode = len(prediction.shape) == 1
        multi_dim_prediction = len(prediction.shape) > 2
        labels_multi_dim = False

        if multi_dim_prediction:
            labels_multi_dim = prediction.shape[2] == 1 and prediction.shape[1] == 1

        if multi_dim_features and prob_prediction:
            new_shape = prediction.shape[0], 1, prediction.shape[1]
            prediction = prediction.reshape(new_shape)
        elif sklearn_output_mode and output_mode != 'labels':
            prediction = prediction.reshape(-1, 1, 1)

        if labels_multi_dim:
            prediction = prediction.squeeze()
        return prediction

    def _convert_to_output(self, prediction,
                           predict_data: InputData,
                           output_data_type: DataTypesEnum = DataTypesEnum.table,
                           output_mode: str = 'default') -> OutputData:
        """Method convert prediction into :obj:`OutputData` if it is not this type yet

        Args:
            prediction: output from model implementation
            predict_data: :obj:`InputData` used for prediction
            output_data_type: :obj:`DataTypesEnum` for output

        Returns: prediction as :obj:`OutputData`
        """
        if type(prediction) is OutputData:
            prediction.predict = self.__convert_dimensions(predict_data, prediction.predict, output_mode)
            converted = prediction
        else:
            prediction = self.__convert_dimensions(predict_data, prediction, output_mode)
            converted = OutputData(idx=predict_data.idx,
                                   features=predict_data.features,
                                   predict=prediction,
                                   task=predict_data.task,
                                   target=predict_data.target,
                                   data_type=output_data_type,
                                   supplementary_data=predict_data.supplementary_data)
        converted.predict = NumpyConverter(data=converted.predict).convert_to_torch_format()
        return converted

    def _sklearn_compatible_prediction(self, trained_operation, predict_data, output_mode: str = 'probs'):
        features = predict_data.features
        if predict_data.task.task_type.value == 'regression':
            return trained_operation.predict(features).flatten()
        else:
            is_multi_output_target = isinstance(trained_operation.classes_, list)
            # Check if target is multilabel (has 2 or more columns)
            if is_multi_output_target:
                n_classes = len(trained_operation.classes_[0])
            else:
                n_classes = len(trained_operation.classes_)
            if output_mode == 'labels':
                prediction = trained_operation.predict(features)
            elif output_mode in ['probs', 'full_probs', 'default']:
                prediction = trained_operation.predict_proba(features)
                if n_classes < 2:
                    raise ValueError('Data set contain only 1 target class. Please reformat your data.')
                elif n_classes == 2 and output_mode != 'full_probs':
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
        if type(predict_data) is not list:
            test_data = [InputData(idx=predict_data.idx,
                                   features=features,
                                   target=predict_data.target,
                                   task=predict_data.task,
                                   data_type=predict_data.data_type,
                                   supplementary_data=predict_data.supplementary_data) for features in
                         predict_data.features.swapaxes(1, 0)]
        else:
            test_data = predict_data
        try:
            prediction = list(
                operation.transform(data).predict for operation, data in zip(trained_operation, test_data))
        except Exception:
            prediction = list(
                operation.transform(data.features) for operation, data in zip(trained_operation, test_data))
        try:
            prediction = np.hstack(prediction)
        except Exception:
            min_dim = min([x.shape[1] for x in prediction])
            prediction = [x[:, :min_dim] for x in prediction]
            prediction = np.stack(prediction).swapaxes(0, 1).squeeze()
        prediction = NumpyConverter(data=prediction).convert_to_torch_format()
        return prediction

    def _custom_fit(self, train_data):
        operation_implementation = self.operation_impl(self.params_for_fit)
        with ImplementationRandomStateHandler(implementation=operation_implementation):
            if type(train_data) is list:
                trained_operation = [deepcopy(operation_implementation) for i in range(len(train_data))]
                fitted_operation = [operation.fit(data) for operation, data in zip(trained_operation, train_data)]
                operation_implementation = fitted_operation
            else:
                operation_implementation = operation_implementation.fit(train_data)
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

    def predict_for_fit(self, trained_operation, predict_data: InputData,
                        mode: str = 'model_fitting', output_mode: str = 'default') -> OutputData:
        """
        Transform method for preprocessing task for fit stage

        Args:
            trained_operation: model object
            predict_data: data used for prediction
        Returns:
            OutputData:
        """
        converted_predict_data = self._convert_input_data(predict_data, mode=mode)
        if mode == 'model_fitting':
            prediction = self._sklearn_compatible_prediction(trained_operation=trained_operation,
                                                             predict_data=converted_predict_data,
                                                             output_mode=output_mode)
        else:
            prediction = self.__operation_multidim_adapter(trained_operation, converted_predict_data)
        converted = self._convert_to_output(prediction, predict_data, predict_data.data_type, output_mode)
        return converted

    def predict(self, trained_operation, predict_data: InputData,
                mode: str = 'model_fitting', output_mode: str = 'default') -> OutputData:
        """Method to predict the target data for predict stage.

        Args:
            trained_operation: trained operation object
            predict_data: data to predict

        Returns:
            passed data with new predicted target
        """
        converted_predict_data = self._convert_input_data(predict_data, mode=mode)

        if type(predict_data) is list:
            data_type = predict_data[0].data_type
            predict_data_copy = predict_data[0]
        else:
            data_type = predict_data.data_type
            predict_data_copy = predict_data

        if mode == 'model_fitting':
            prediction = self._sklearn_compatible_prediction(trained_operation=trained_operation,
                                                             predict_data=converted_predict_data,
                                                             output_mode=output_mode)
        else:
            prediction = self.__operation_multidim_adapter(trained_operation, predict_data)

        converted = self._convert_to_output(prediction, predict_data_copy, data_type, output_mode)
        return converted


class IndustrialCustomPreprocessingStrategy:
    _operations_by_types = FEDOT_PREPROC_MODEL

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        self.multi_dim_dispatcher = MultiDimPreprocessingStrategy(self.operation_impl, operation_type)
        self.params_for_fit = params or OperationParameters()
        self.operation_id = operation_type
        self.output_mode = False

    @property
    def operation_type(self):
        return get_operation_type_from_id(self.operation_id)

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self._operations_by_types:
            return self._operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain {self.__class__} strategy for {operation_type}')

    def fit(self, train_data: InputData):
        return self.multi_dim_dispatcher.fit(train_data, mode='custom_fit')

    def predict(self, trained_operation, predict_data: InputData, output_mode: str = 'default'):
        return self.multi_dim_dispatcher.predict(trained_operation, predict_data,
                                                 mode='feature_extraction', output_mode=output_mode)

    def predict_for_fit(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        return self.multi_dim_dispatcher.predict_for_fit(trained_operation, predict_data,
                                                         mode='feature_extraction', output_mode=output_mode)


class IndustrialPreprocessingStrategy(IndustrialCustomPreprocessingStrategy):
    __operations_by_types = INDUSTRIAL_PREPROC_MODEL

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        params = IndustrialOperationParameters().from_params(operation_type, params) if params \
            else IndustrialOperationParameters().from_operation_type(operation_type)
        self.multi_dim_dispatcher = MultiDimPreprocessingStrategy(self.operation_impl, operation_type)
        super().__init__(operation_type, params)

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain custom preprocessing strategy for {operation_type}')

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
            operation_implementation.fit(train_data)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        """Transform method for preprocessing task

        Args:
            trained_operation: model object
            predict_data: data used for prediction

        Returns:
            prediction
            :param output_mode:
        """
        prediction = trained_operation.transform(predict_data)
        converted = self.multi_dim_dispatcher._convert_to_output(prediction, predict_data)
        return converted

    def predict_for_fit(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        """
        Transform method for preprocessing task for fit stage

        Args:
            trained_operation: model object
            predict_data: data used for prediction
        Returns:
            OutputData:
        """
        prediction = trained_operation.transform_for_fit(predict_data)
        converted = self.multi_dim_dispatcher._convert_to_output(prediction, predict_data)
        return converted


class IndustrialClassificationPreprocessingStrategy(IndustrialCustomPreprocessingStrategy):
    """ Strategy for applying custom algorithms from FEDOT to preprocess data
    for classification task
    """

    _operations_by_types = INDUSTRIAL_CLF_PREPROC_MODEL

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)
        self.multi_dim_dispatcher = MultiDimPreprocessingStrategy(self.operation_impl, operation_type)

    def fit(self, train_data: InputData):
        return self.multi_dim_dispatcher.fit(train_data, mode='feature_extraction')


class IndustrialSkLearnEvaluationStrategy(IndustrialCustomPreprocessingStrategy):

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        self.multi_dim_dispatcher = MultiDimPreprocessingStrategy(self.operation_impl, operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        return self.multi_dim_dispatcher.fit(train_data, mode='model_fitting')

    def predict(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        return self.multi_dim_dispatcher.predict(trained_operation, predict_data,
                                                 mode='model_fitting', output_mode=output_mode)

    def predict_for_fit(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        return self.multi_dim_dispatcher.predict_for_fit(trained_operation, predict_data,
                                                         mode='model_fitting', output_mode=output_mode)


class IndustrialSkLearnClassificationStrategy(IndustrialSkLearnEvaluationStrategy):
    """ Strategy for applying classification algorithms from Sklearn library """
    _operations_by_types = SKLEARN_CLF_MODELS

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        self.multi_dim_dispatcher = MultiDimPreprocessingStrategy(self.operation_impl, operation_type)
        super().__init__(operation_type, params)


class IndustrialSkLearnRegressionStrategy(IndustrialSkLearnEvaluationStrategy):
    """ Strategy for applying regression algorithms from Sklearn library """
    _operations_by_types = SKLEARN_REG_MODELS

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        self.multi_dim_dispatcher = MultiDimPreprocessingStrategy(self.operation_impl, operation_type)
        super().__init__(operation_type, params)

    def predict(self, trained_operation, predict_data: InputData, output_mode: str = 'labels') -> OutputData:
        return self.multi_dim_dispatcher.predict(trained_operation, predict_data,
                                                 mode='model_fitting', output_mode='labels')

    def predict_for_fit(self, trained_operation, predict_data: InputData, output_mode: str = 'labels') -> OutputData:
        return self.multi_dim_dispatcher.predict_for_fit(trained_operation, predict_data,
                                                         mode='model_fitting', output_mode='labels')


class IndustrialCustomRegressionStrategy(IndustrialSkLearnEvaluationStrategy):
    _operations_by_types = SKLEARN_REG_MODELS

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        self.multi_dim_dispatcher = MultiDimPreprocessingStrategy(self.operation_impl, operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        train_data = self.multi_dim_dispatcher._convert_input_data(train_data, mode='model_fitting')
        return self.multi_dim_dispatcher.fit(train_data, mode='custom_fit')


class IndustrialDataSourceStrategy(IndustrialCustomPreprocessingStrategy):

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        return object()

    def predict(self, trained_operation, predict_data: InputData, output_mode: str = 'labels') -> OutputData:
        return OutputData(idx=predict_data.idx, features=predict_data.features, task=predict_data.task,
                          data_type=predict_data.data_type, target=predict_data.target, predict=predict_data.features)

    def _convert_to_operation(self, operation_type: str):
        return object()
