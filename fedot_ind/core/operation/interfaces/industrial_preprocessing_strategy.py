import warnings
from copy import deepcopy
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy, \
    convert_to_multivariate_model, is_multi_output_task
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import \
    *
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operation_type_from_id
from fedot.utilities.random import ImplementationRandomStateHandler

from fedot_ind.core.architecture.preprocessing.data_convertor import NumpyConverter
from fedot_ind.core.repository.model_repository import INDUSTRIAL_CLF_PREPROC_MODEL, FEDOT_PREPROC_MODEL, \
    INDUSTRIAL_PREPROC_MODEL
from fedot_ind.core.repository.IndustrialOperationParameters import IndustrialOperationParameters


class MultiDimPreprocessingStrategy(EvaluationStrategy):
    def __init__(self, operation_impl,
                 operation_type: str, params: Optional[OperationParameters] = None,
                 mode: str = 'one_dimensional'):
        self.operation_impl = operation_impl
        super().__init__(operation_type, params)
        self.output_mode = 'labels'
        self.mode = mode

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))

    def __convert_input_data(self, input_data):
        if len(input_data.features.shape) > 2:
            converted_data = self._convert_input_data(input_data)
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

    def _convert_to_output(self,
                           prediction,
                           predict_data: InputData,
                           output_data_type: DataTypesEnum = DataTypesEnum.table,
                           output_mode: str = 'default') -> OutputData:
        if type(prediction) is OutputData:
            converted = prediction
        else:
            converted = OutputData(idx=predict_data.idx,
                                   features=predict_data.features,
                                   predict=prediction,
                                   task=predict_data.task,
                                   target=predict_data.target,
                                   data_type=output_data_type,
                                   supplementary_data=predict_data.supplementary_data)
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
                prediction = trained_operation.predict(features).reshape(-1,1)
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

    def _convert_input_data(self, train_data):
        if self.mode == 'one_dimensional':
            new_shape = train_data.features.shape[0], train_data.features.shape[1] * train_data.features.shape[2]
            features = train_data.features.reshape(new_shape)
            input_data = InputData(idx=train_data.idx,
                                   features=features,
                                   target=train_data.target,
                                   task=train_data.task,
                                   data_type=train_data.data_type,
                                   supplementary_data=train_data.supplementary_data)
        elif self.mode == 'channel_independent':
            input_data = [InputData(idx=train_data.idx,
                                    features=features,
                                    target=train_data.target,
                                    task=train_data.task,
                                    data_type=train_data.data_type,
                                    supplementary_data=train_data.supplementary_data) for features in
                          train_data.features.swapaxes(1, 0)]
        elif self.mode == 'multi_dimensional':
            features = NumpyConverter(data=train_data.features).convert_to_torch_format()
            input_data = InputData(idx=train_data.idx,
                                   features=features,
                                   target=train_data.target,
                                   task=train_data.task,
                                   data_type=train_data.data_type,
                                   supplementary_data=train_data.supplementary_data)
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
        operation_implementation.fit(train_data)
        return operation_implementation

    def fit_one_sample(self, train_data: InputData):
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

    def fit(self, train_data: InputData):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        train_data = self._convert_input_data(train_data)

        if self.mode == 'one_dimensional':
            return self.fit_one_sample(train_data)
        elif self.mode == 'channel_independent':
            operation_implementation = self.operation_impl(self.params_for_fit)
            with ImplementationRandomStateHandler(implementation=operation_implementation):
                trained_operation = [deepcopy(operation_implementation) for i in range(len(train_data))]
                fitted_operation = [operation.fit(data) for operation, data in zip(trained_operation, train_data)]
                operation_implementation = fitted_operation
            return operation_implementation
        else:
            return self._custom_fit(train_data)

    def predict_for_fit(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        converted_predict_data = self._convert_input_data(predict_data)
        if self.mode == 'one_dimensional':
            prediction = self._sklearn_compatible_prediction(trained_operation=trained_operation,
                                                             predict_data=converted_predict_data,
                                                             output_mode=output_mode)
        elif self.mode == 'channel_independent':
            prediction = self.__operation_multidim_adapter(trained_operation, converted_predict_data)
        elif self.mode == 'multi_dimensional':
            if 'predict_for_fit' in dir(trained_operation):
                prediction = trained_operation.predict_for_fit(predict_data, output_mode)
            else:
                prediction = trained_operation.predict(predict_data, output_mode)
        converted = self._convert_to_output(prediction, predict_data, predict_data.data_type, output_mode)
        return converted

    def predict(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        converted_predict_data = self._convert_input_data(predict_data)

        if type(predict_data) is list:
            data_type = predict_data[0].data_type
            predict_data_copy = predict_data[0]
        else:
            data_type = predict_data.data_type
            predict_data_copy = predict_data

        if self.mode == 'one_dimensional':
            prediction = self._sklearn_compatible_prediction(trained_operation=trained_operation,
                                                             predict_data=converted_predict_data,
                                                             output_mode=output_mode)
        elif self.mode == 'channel_independent':
            prediction = self.__operation_multidim_adapter(trained_operation, converted_predict_data)
        elif self.mode == 'multi_dimensional':
            prediction = trained_operation.predict(predict_data, output_mode)

        converted = self._convert_to_output(prediction, predict_data_copy, data_type, output_mode)
        return converted


class IndustrialCustomPreprocessingStrategy:
    _operations_by_types = FEDOT_PREPROC_MODEL

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        self.multi_dim_dispatcher = MultiDimPreprocessingStrategy(self.operation_impl,
                                                                  operation_type,
                                                                  mode='channel_independent')
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
        return self.multi_dim_dispatcher.fit(train_data)

    def predict(self, trained_operation, predict_data: InputData, output_mode: str = 'default'):
        return self.multi_dim_dispatcher.predict(trained_operation, predict_data, output_mode=output_mode)

    def predict_for_fit(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        return self.multi_dim_dispatcher.predict_for_fit(trained_operation, predict_data, output_mode=output_mode)


class IndustrialPreprocessingStrategy(IndustrialCustomPreprocessingStrategy):
    __operations_by_types = INDUSTRIAL_PREPROC_MODEL

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        params = IndustrialOperationParameters().from_params(operation_type, params) if params \
            else IndustrialOperationParameters().from_operation_type(operation_type)
        super().__init__(operation_type, params)

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain custom preprocessing strategy for {operation_type}')

    def fit(self, train_data: InputData):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        operation_implementation = self.operation_impl(self.params_for_fit)
        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        prediction = trained_operation.transform(predict_data)
        converted = self.multi_dim_dispatcher._convert_to_output(prediction, predict_data)
        return converted

    def predict_for_fit(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        prediction = trained_operation.transform_for_fit(predict_data)
        converted = self.multi_dim_dispatcher._convert_to_output(prediction, predict_data)
        return converted


class IndustrialClassificationPreprocessingStrategy(IndustrialCustomPreprocessingStrategy):
    _operations_by_types = INDUSTRIAL_CLF_PREPROC_MODEL

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        return self.multi_dim_dispatcher.fit(train_data)


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

    def predict_for_fit(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        return OutputData(idx=predict_data.idx, features=predict_data.features, task=predict_data.task,
                          data_type=predict_data.data_type, target=predict_data.target, predict=predict_data.features)
