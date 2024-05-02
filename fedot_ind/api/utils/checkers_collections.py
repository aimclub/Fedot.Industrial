import logging
from typing import Union

import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TsForecastingParams, TaskTypesEnum
from sklearn.preprocessing import LabelEncoder

from fedot_ind.api.utils.data import check_multivariate_data
from fedot_ind.core.architecture.preprocessing.data_convertor import NumpyConverter, DataConverter
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.repository.constanst_repository import FEDOT_TASK


class DataCheck:
    """Class for checking and preprocessing input data for Fedot AutoML.

    Args:
        input_data: Input data in tuple format (X, y) or Fedot InputData object.
        task: Machine learning task, either "classification" or "regression".

    Attributes:
        logger (logging.Logger): Logger instance for logging messages.
        input_data (InputData): Preprocessed and initialized Fedot InputData object.
        task (str): Machine learning task for the dataset.
        task_dict (dict): Mapping of string task names to Fedot Task objects.

    """

    def __init__(self,
                 input_data: Union[tuple, InputData] = None,
                 task: str = None,
                 task_params=None,
                 industrial_task_params=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.industrial_task_params = industrial_task_params
        self.input_data = input_data
        self.data_convertor = DataConverter(data=self.input_data)
        self.task = task
        self.task_params = task_params
        self.task_dict = FEDOT_TASK
        self.label_encoder = None

    def __check_features_and_target(self, X, y):
        multi_features, X = check_multivariate_data(X)
        multi_target = len(y.shape) > 1 and y.shape[1] > 2

        if multi_features:
            features = np.array(X.tolist()).astype(float)
        else:
            features = X

        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        if multi_target:
            target = y
        elif multi_features and not multi_target:
            target = y.reshape(-1, 1)
        else:
            target = np.ravel(y).reshape(-1, 1)

        return features, multi_features, target

    def _init_input_data(self) -> None:
        """Initializes the `input_data` attribute based on its type.

        If a tuple (X, y) is provided, it converts it to a Fedot InputData object
        with appropriate data types and task information. If an existing InputData
        object is provided, it checks if it requires further initialization.

        Raises:
            ValueError: If the input data format is invalid.

        """
        # is_multivariate_data = False

        if self.data_convertor.is_tuple:
            features, is_multivariate_data, target = self.__check_features_and_target(self.input_data[0],
                                                                                      self.input_data[1])
        else:
            features, is_multivariate_data, target = self.__check_features_and_target(self.input_data.features,
                                                                                      self.input_data.target)

        if self.label_encoder is None and self.task == 'classification':
            # x, y = self.input_data.features, self.input_data.target
            if type(target[0]) is np.str_:
                self.label_encoder = LabelEncoder()
                target = self.label_encoder.fit_transform(target)
            # else:
            #     self.label_encoder = self.label_encoder

        if is_multivariate_data:
            self.input_data = InputData(idx=np.arange(len(features)),
                                        features=features,
                                        target=target,
                                        task=self.task_dict[self.task],
                                        data_type=DataTypesEnum.image)
        elif self.task == 'ts_forecasting':
            features_array = self.data_convertor.convert_to_1d_array()
            task = Task(TaskTypesEnum.ts_forecasting,
                        TsForecastingParams(forecast_length=self.task_params['forecast_length']))
            if self.industrial_task_params is None:
                features_array = features_array[:-self.task_params['forecast_length']]
                target = features_array
            self.input_data = InputData.from_numpy_time_series(
                features_array=features_array,
                target_array=target,
                task=task
            )
        else:
            self.input_data = InputData(idx=np.arange(len(features)),
                                        features=features,
                                        target=target,
                                        task=self.task_dict[self.task],
                                        data_type=DataTypesEnum.image)

    def _check_input_data_features(self):
        """Checks and preprocesses the features in the input data.

        - Replaces NaN and infinite values with 0.
        - Converts features to torch format using NumpyConverter.

        """
        self.input_data.features = np.where(
            np.isnan(self.input_data.features), 0, self.input_data.features)
        self.input_data.features = np.where(
            np.isinf(self.input_data.features), 0, self.input_data.features)
        if self.task != 'ts_forecasting':
            self.input_data.features = NumpyConverter(
                data=self.input_data.features).convert_to_torch_format()

    def _check_input_data_target(self):
        """Checks and preprocesses the features in the input data.

        - Replaces NaN and infinite values with 0.
        - Converts features to torch format using NumpyConverter.

        """
        if self.input_data.target is not None and type(
                self.input_data.target.ravel()[0]) is np.str_ and self.task == 'regression':
            self.input_data.target = self.input_data.target.astype(float)

        elif self.task == 'regression':
            self.input_data.target = self.input_data.target.squeeze()
        elif self.task == 'classification':
            self.input_data.target[self.input_data.target == -1] = 0

    def check_available_operations(self, available_operations):
        pass

    def check_input_data(self) -> InputData:
        """Checks and preprocesses the input data for Fedot AutoML.

        Performs the following steps:
            1. Initializes the `input_data` attribute based on its type.
            2. Checks and preprocesses the features (replacing NaNs, converting to torch format).
            3. Checks and preprocesses the target variable (encoding labels, casting to float).

        Returns:
            InputData: The preprocessed and initialized Fedot InputData object.

        """

        self._init_input_data()
        self._check_input_data_features()
        self._check_input_data_target()
        return self.input_data

    def get_target_encoder(self):
        return self.label_encoder
