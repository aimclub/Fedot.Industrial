from typing import Optional

from fedot.api.main import Fedot
from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot_ind.integration.fedot.compatibility import as_numpy, ensure_fedot_tensor_data
from fedot_ind.integration.fedot.extensions import industrial_extension_scope
from fedot_ind.integration.fedot.extensions.discovery import default_industrial_availiable_operation


class FedotAutomlImplementation(ModelImplementation):
    """Implementation of Fedot as classification pipeline node for AutoML.

    """
    AVAILABLE_OPERATIONS = default_industrial_availiable_operation(
        'classification')

    def __init__(self, params: Optional[OperationParameters] = None):
        if not params:
            params = OperationParameters()
        else:
            params = params.to_dict()
        if 'available_operations' not in params.keys():
            params.update({'available_operations': self.AVAILABLE_OPERATIONS})
        if 'initial_assumption' not in params:
            task = params.get('problem', 'classification')
            operation = 'pdl_reg' if task == 'regression' else 'pdl_clf'
            with industrial_extension_scope():
                params['initial_assumption'] = PipelineBuilder().add_node(operation).build()
        with industrial_extension_scope():
            self.model = Fedot(**params)
        self._fedot_train_data = None
        super(FedotAutomlImplementation, self).__init__()

    def fit(self, input_data: InputData):
        self._fedot_train_data = ensure_fedot_tensor_data(input_data, fit_stage=True)
        with industrial_extension_scope():
            self.model.fit(self._fedot_train_data)
        return self

    def predict(
            self,
            input_data: InputData,
            output_mode='default') -> OutputData:
        tensor_data = ensure_fedot_tensor_data(
            input_data,
            fit_stage=False,
            reference_data=self._fedot_train_data,
        )
        with industrial_extension_scope():
            prediction = self.model.current_pipeline.predict(
                tensor_data, output_mode=output_mode)
        return as_numpy(prediction.predict)


class FedotClassificationImplementation(FedotAutomlImplementation):
    """Implementation of Fedot as classification pipeline node for AutoML.

    """
    AVAILABLE_OPERATIONS = default_industrial_availiable_operation(
        'classification')


class FedotRegressionImplementation(FedotAutomlImplementation):
    """Implementation of Fedot as regression pipeline node for AutoML.

    """
    AVAILABLE_OPERATIONS = default_industrial_availiable_operation(
        'regression')


class FedotForecastingImplementation(FedotAutomlImplementation):
    """Implementation of Fedot as forecasting pipeline node for AutoML.

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        self.model = Fedot
        self._fedot_train_data = None
        self.metric = params.get('metric', 'mape')
        self.timeout = params.get('timeout', 5)
        self.finetune = params.get('with_tuning', True)
        self.available_operations = ['ar',
                                     'gaussian_filter',
                                     'lagged',
                                     'lasso',
                                     'rfr',
                                     'ridge',
                                     'sgdr',
                                     'smoothing',
                                     'sparse_lagged',
                                     'svr'
                                     ]

    def fit(self, input_data: InputData):
        self._fedot_train_data = ensure_fedot_tensor_data(input_data, fit_stage=True)
        self.model = self.model(task_params=input_data.task.task_params,
                                problem='ts_forecasting',
                                available_operations=self.available_operations,
                                metric=self.metric,
                                with_tuning=self.finetune,
                                logging_level=30,
                                timeout=self.timeout)
        with industrial_extension_scope():
            self.model.fit(self._fedot_train_data)
        self.model = self.model.current_pipeline
        return self

    def predict(
            self,
            input_data: InputData,
            output_mode='default') -> OutputData:
        tensor_data = ensure_fedot_tensor_data(
            input_data,
            fit_stage=False,
            reference_data=self._fedot_train_data,
        )
        with industrial_extension_scope():
            prediction = self.model.predict(tensor_data)
        return as_numpy(prediction.predict)
