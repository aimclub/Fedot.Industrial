import warnings
from copy import deepcopy
from inspect import signature
from typing import Optional

import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import convert_to_multivariate_model, EvaluationStrategy, \
    is_multi_output_task
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import get_operation_type_from_id, OperationTypesRepository
from fedot.utilities.random import ImplementationRandomStateHandler

from fedot_ind.core.architecture.preprocessing.data_convertor import ConditionConverter, FedotConverter, NumpyConverter
from fedot_ind.core.repository.IndustrialOperationParameters import IndustrialOperationParameters
from fedot_ind.core.repository.model_repository import FEDOT_PREPROC_MODEL, FORECASTING_PREPROC, \
    INDUSTRIAL_CLF_PREPROC_MODEL, INDUSTRIAL_PREPROC_MODEL


class MultiDimPreprocessingStrategy(EvaluationStrategy):
    def __init__(self, operation_impl,
                 operation_type: str,
                 params: Optional[OperationParameters] = None,
                 mode: str = 'one_dimensional'
                 ):
        self.operation_impl = operation_impl
        super().__init__(operation_type, params)
        self.output_mode = 'labels'
        self.concat_func = np.hstack
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
        if not self.operation_condition.is_operation_is_list_container:
            prediction = trained_operation.transform(predict_data) if self.operation_condition.is_transform_input_fedot \
                else trained_operation.transform(predict_data.features)
        elif self.operation_condition.have_predict_atr and self.operation_condition.is_operation_is_list_container:
            prediction = trained_operation
        else:
            prediction = self._predict_for_ndim(predict_data, trained_operation)
        return prediction

    def _convert_to_output(self,
                           prediction,
                           predict_data: InputData,
                           output_data_type: DataTypesEnum = DataTypesEnum.table,
                           output_mode: str = 'default') -> OutputData:

        return FedotConverter(data=predict_data).convert_to_output_data(prediction,
                                                                        predict_data,
                                                                        output_data_type)

    def _sklearn_compatible_prediction(self, trained_operation, predict_data, output_mode: str = 'probs'):
        if self.operation_condition.is_regression_of_forecasting_task:
            return trained_operation.predict(predict_data).predict if self.operation_condition.is_predict_input_fedot \
                else trained_operation.predict(predict_data.features).flatten()
        else:
            n_classes = len(trained_operation.classes_[0]) if self.operation_condition.is_multi_output_target \
                else len(trained_operation.classes_)
            prediction = self.operation_condition.output_mode_converter(output_mode, n_classes)
            return prediction

    def _convert_input_data(self, train_data, mode: str = None):
        return FedotConverter(train_data).convert_to_industrial_composing_format(mode if mode is not None
                                                                                 else self.mode)

    def _predict_for_ndim(self, predict_data, trained_operation: list):
        self.operation_condition_for_channel_independent = ConditionConverter(predict_data,
                                                                              trained_operation[0],
                                                                              self.mode)

        # create list of InputData, where each InputData correspond to each channel
        test_data = predict_data if type(predict_data) is list else \
            [InputData(idx=predict_data.idx,
                       features=features,
                       target=predict_data.target,
                       task=predict_data.task,
                       data_type=predict_data.data_type,
                       supplementary_data=predict_data.supplementary_data) for features in
             predict_data.features.swapaxes(1, 0)]

        # check model methods and method input type

        if self.operation_condition_for_channel_independent.have_transform_method:
            if self.operation_condition_for_channel_independent.is_transform_input_fedot:
                prediction = list(operation.transform(
                    data) for operation, data in zip(trained_operation, test_data))
                if self.operation_type == 'lagged' or self.operation_type == 'sparse_lagged':
                    prediction = prediction
                else:
                    prediction = [pred.predict if type(
                        pred) is not np.ndarray else pred for pred in prediction]
            else:
                prediction = list(
                    operation.transform(data.features) for operation, data in zip(trained_operation, test_data))
        elif self.operation_condition_for_channel_independent.have_predict_method:
            prediction = list(operation.predict(data)
                              for operation, data in zip(trained_operation, test_data))

            prediction = [pred.predict for pred in prediction if type(pred) is not np.array]

        if not isinstance(prediction[0], OutputData):
            prediction = NumpyConverter(data=self.concat_func(prediction)).convert_to_torch_format()
        return prediction

    def _custom_fit(self, train_data):
        operation_implementation = self.operation_impl(self.params_for_fit)
        operation_implementation.fit(train_data)
        return operation_implementation

    def fit_one_sample(self, operation_implementation, train_data: InputData):

        # If model doesn't support multi-output and current task is ts_forecasting
        current_task = train_data.task.task_type
        models_repo = OperationTypesRepository()
        non_multi_models = models_repo.suitable_operation(task_type=current_task,
                                                          tags=['non_multi'])
        is_model_not_support_multi = self.operation_type in non_multi_models

        # Multi-output task or not
        is_multi_target = is_multi_output_task(train_data)

        if is_model_not_support_multi and is_multi_target:
            # Manually wrap the regressor into multi-output model
            operation_implementation = convert_to_multivariate_model(operation_implementation,
                                                                     train_data)
        else:
            sig = signature(operation_implementation.fit).parameters
            if len(sig) > 1:
                operation_implementation.fit(train_data.features, train_data.target)
            else:
                operation_implementation.fit(train_data)
        return operation_implementation

    def _init_impl(self, channel_params):
        try:
            operation_implementation = self.operation_impl(**channel_params.to_dict())
        except Exception:
            operation_implementation = self.operation_impl(channel_params)
        return operation_implementation

    def fit(self, train_data: InputData):
        # init operation_impl model abstraction
        if isinstance(self.params_for_fit, list):
            operation_implementation = [self._init_impl(channel_params) for channel_params in self.params_for_fit]
        else:
            operation_implementation = self._init_impl(self.params_for_fit)

        # Create model and data condition checker
        self.operation_condition = ConditionConverter(train_data, operation_implementation, self.mode)
        # If model is classical sklearn model we use one_dimensional mode
        if self.operation_condition.is_one_dim_operation:
            return self.fit_one_sample(operation_implementation, train_data)
        # Elif model could be use for each dimension(channel) independently we use channel_independent mode
        elif self.operation_condition.is_channel_independent_operation:
            # Create independent copy of model for each channel
            if self.operation_condition.is_operation_is_list_container:
                trained_operation = operation_implementation
            else:
                trained_operation = [deepcopy(operation_implementation) if self.operation_condition.is_list_container
                                     else deepcopy(operation_implementation) for i in range(len(train_data))]

            train_data = train_data if self.operation_condition.is_list_container else [train_data]

            # Check if model have both or just one method (fit and transform_for_fit). For some model one of this method
            # could be not finished to use right now.
            if self.operation_condition.have_fit_method:
                operation_implementation = [operation.fit(data) for operation, data in zip(
                    trained_operation, train_data)]

                if not type(operation_implementation[0]) == type(trained_operation[0]):
                    operation_implementation = trained_operation
                fit_method_is_not_implemented = operation_implementation[0] is None
            elif self.operation_condition.have_transform_method:
                operation_implementation = [operation.transform_for_fit(data) for operation, data in zip(
                    trained_operation, train_data)]

            if fit_method_is_not_implemented:
                operation_implementation = [operation.transform_for_fit(data) for operation, data in zip(
                    trained_operation, train_data)]

            return operation_implementation
        else:
            return self._custom_fit(train_data)

    def predict_for_fit(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        data_type, predict_data_copy = FedotConverter(predict_data).unwrap_list_to_output()
        # Create data condition checker
        self.operation_condition = ConditionConverter(predict_data, trained_operation, self.mode)

        if self.operation_condition.is_one_dim_operation:
            if self.operation_condition.have_predict_for_fit_method:
                # try except because some model implementation  have a different logic in predict
                # and predict for fit methods and it leads to errors
                try:
                    prediction = trained_operation.predict_for_fit(predict_data)
                except Exception:
                    prediction = self._sklearn_compatible_prediction(trained_operation=trained_operation,
                                                                     predict_data=predict_data,
                                                                     output_mode=output_mode)
            else:
                prediction = self._sklearn_compatible_prediction(trained_operation=trained_operation,
                                                                 predict_data=predict_data,
                                                                 output_mode=output_mode)
        elif self.operation_condition.is_channel_independent_operation:
            prediction = self.__operation_multidim_adapter(trained_operation, predict_data)
        elif self.operation_condition.is_multi_dimensional_operation:
            if self.operation_condition.have_predict_for_fit_method:
                prediction = trained_operation.predict_for_fit(
                    predict_data, output_mode)
            else:
                prediction = trained_operation.predict(
                    predict_data, output_mode)

        converted = self._convert_to_output(prediction, predict_data_copy, data_type, output_mode)
        return converted

    def predict(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        data_type, predict_data_copy = FedotConverter(
            predict_data).unwrap_list_to_output()

        # Create data condition checker
        self.operation_condition = ConditionConverter(
            predict_data, trained_operation, self.mode)
        if self.operation_condition.is_one_dim_operation:
            prediction = self._sklearn_compatible_prediction(trained_operation=trained_operation,
                                                             predict_data=predict_data,
                                                             output_mode=output_mode)
        elif self.operation_condition.is_channel_independent_operation:
            prediction = self.__operation_multidim_adapter(
                trained_operation, predict_data)
        elif self.operation_condition.is_multi_dimensional_operation:
            prediction = trained_operation.predict(predict_data, output_mode)

        converted = self._convert_to_output(
            prediction, predict_data_copy, data_type, output_mode)
        return converted


class IndustrialCustomPreprocessingStrategy:
    _operations_by_types = FEDOT_PREPROC_MODEL

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        if params is None or operation_type == 'xgboost':
            params = IndustrialOperationParameters().from_operation_type(operation_type)
        else:
            params = IndustrialOperationParameters().from_params(operation_type, params)
        self.multi_dim_dispatcher = MultiDimPreprocessingStrategy(self.operation_impl,
                                                                  operation_type,
                                                                  params=params,
                                                                  mode='channel_independent')
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
            raise ValueError(
                f'Impossible to obtain {self.__class__} strategy for {operation_type}')

    def fit(self, train_data: InputData):
        train_data = self.multi_dim_dispatcher._convert_input_data(train_data)
        return self.multi_dim_dispatcher.fit(train_data)

    def predict(self, trained_operation, predict_data: InputData, output_mode: str = 'default'):
        converted_predict_data = self.multi_dim_dispatcher._convert_input_data(predict_data)
        return self.multi_dim_dispatcher.predict(trained_operation, converted_predict_data, output_mode=output_mode)

    def predict_for_fit(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        converted_predict_data = self.multi_dim_dispatcher._convert_input_data(
            predict_data)
        return self.multi_dim_dispatcher.predict_for_fit(trained_operation, converted_predict_data,
                                                         output_mode=output_mode)


class IndustrialPreprocessingStrategy(IndustrialCustomPreprocessingStrategy):
    _operations_by_types = INDUSTRIAL_PREPROC_MODEL

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        params = IndustrialOperationParameters().from_params(operation_type, params) if params \
            else IndustrialOperationParameters().from_operation_type(operation_type)
        super().__init__(operation_type, params)
        self.params_for_fit = self.multi_dim_dispatcher.params_for_fit

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self._operations_by_types.keys():
            return self._operations_by_types[operation_type]
        else:
            raise ValueError(
                f'Impossible to obtain custom preprocessing strategy for {operation_type}')

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


class IndustrialForecastingPreprocessingStrategy(IndustrialCustomPreprocessingStrategy):
    _operations_by_types = FORECASTING_PREPROC

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        params = IndustrialOperationParameters().from_params(operation_type, params) if params \
            else IndustrialOperationParameters().from_operation_type(operation_type)
        super().__init__(operation_type, params)
        self.multi_dim_dispatcher.concat_func = np.vstack
        self.ensemble_func = np.sum

    def _check_exog_params(self, fit_output):
        if self.operation_type == 'exog_ts':
            for output in fit_output:
                output.supplementary_data.is_main_target = False
        return fit_output

    def fit(self, train_data: InputData):
        train_data = self.multi_dim_dispatcher._convert_input_data(train_data)
        fit_output = self.multi_dim_dispatcher.fit(train_data)
        fit_output = self._check_exog_params(fit_output)
        return fit_output

    def predict(self, trained_operation,
                predict_data: InputData,
                output_mode: str = 'default'):
        converted_predict_data = self.multi_dim_dispatcher._convert_input_data(predict_data)
        predict_output = self.multi_dim_dispatcher.predict(trained_operation, converted_predict_data,
                                                           output_mode=output_mode)
        return predict_output

    def predict_for_fit(self, trained_operation,
                        predict_data: InputData,
                        output_mode: str = 'default') -> OutputData:
        converted_predict_data = self.multi_dim_dispatcher._convert_input_data(predict_data)
        predict_output = self.multi_dim_dispatcher.predict_for_fit(trained_operation,
                                                                   converted_predict_data,
                                                                   output_mode=output_mode)
        return predict_output


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
        return FedotConverter(predict_data).convert_input_to_output()

    def _convert_to_operation(self, operation_type: str):
        return object()

    def predict_for_fit(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        return FedotConverter(predict_data).convert_input_to_output()
