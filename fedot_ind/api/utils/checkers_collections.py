import logging
from typing import Union

import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TsForecastingParams, TaskTypesEnum
from pymonad.either import Either
from pymonad.list import ListMonad
from sklearn.preprocessing import LabelEncoder

from fedot_ind.api.utils.data import check_multivariate_data
from fedot_ind.core.architecture.preprocessing.data_convertor import NumpyConverter, DataConverter
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.repository.constanst_repository import FEDOT_DATA_TYPE, fedot_task


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

        if self.industrial_task_params is not None:
            self.data_type = FEDOT_DATA_TYPE[self.industrial_task_params['data_type']]
        else:
            self.data_type = FEDOT_DATA_TYPE['tensor']

        self.input_data = input_data
        self.data_convertor = DataConverter(data=self.input_data)

        self.task = task
        self.task_params = task_params
        self.label_encoder = None

    def __check_features_and_target(self, input_data, data_type):
        if data_type == 'torchvision':
            X, multi_features, y = input_data[0].data.cpu().detach(
            ).numpy(), True, input_data[0].targets.cpu().detach().numpy()
        elif self.data_convertor.is_tuple:
            X, y = input_data[0], input_data[1]
        else:
            X, y = input_data.features, input_data.target

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

    def _encode_target(self, data_tuple):
        self.label_encoder = LabelEncoder()
        data_tuple[1] = self.label_encoder.fit_transform(data_tuple[1])
        return data_tuple[1]

    def _transformation_for_ts_forecasting(self):
        if self.data_convertor.is_numpy_matrix and any(
                [self.data_convertor.have_one_sample, self.data_convertor.have_one_channel]):
            features_array = self.data_convertor.convert_to_1d_array()
        else:
            features_array = self.data_convertor.numpy_data
        task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(
            forecast_length=self.task_params['forecast_length']))
        if self.industrial_task_params is None:
            features_array = features_array[:-
            self.task_params['forecast_length']]
            target = features_array
        return InputData.from_numpy_time_series(
            features_array=features_array, target_array=target, task=task)

    def _transformation_for_other_task(self, data_list):
        encoder_condition = all([self.label_encoder is None,
                                 self.task == 'classification',
                                 isinstance(data_list[1][0],
                                            np.str_)])
        input_data = Either(
            value=data_list,
            monoid=[
                data_list,
                encoder_condition]).either(
            left_function=lambda l: ListMonad(l),
            right_function=ListMonad(
                self._encode_target)).value[0]
        idx = Either(
            value=input_data, monoid=[
                data_list, not self.data_convertor.is_torchvision_dataset]).either(
            left_function=lambda l: np.arange(
                l[0].shape[0]), right_function=lambda r: np.arange(
                len(
                    data_list[0])))

        have_predict_horizon = all([self.industrial_task_params is not None,
                                    self.industrial_task_params['data_type'] == 'time_series',
                                    'detection_window' in self.industrial_task_params.keys()])
        task = Either(
            value=fedot_task(self.task), monoid=[
                fedot_task('ts_forecasting', self.industrial_task_params['detection_window']),
                not have_predict_horizon]).either(left_function=lambda l: l, right_function=lambda r: r)
        return InputData(idx=idx,
                         features=input_data[0],
                         target=input_data[1],
                         task=task,
                         data_type=self.data_type)

    def _init_input_data(self) -> None:
        """Initializes the `input_data` attribute based on its type.

        If a tuple (X, y) is provided, it converts it to a Fedot InputData object
        with appropriate data types and task information. If an existing InputData
        object is provided, it checks if it requires further initialization.

        Raises:
            ValueError: If the input data format is invalid.

        """
        # is_multivariate_data = False
        features, self.is_multivariate_data, target = Either(value=self.input_data,
                                                             monoid=[self.data_convertor,
                                                                     self.data_convertor.is_tuple]).either(
            right_function=lambda r: ListMonad('tuple'),
            left_function=lambda l: ListMonad('torchvision')). \
            then(lambda data_type: self.__check_features_and_target(self.input_data, data_type))

        self.input_data = Either(
            value=[
                features,
                target],
            monoid=[
                [
                    features,
                    target],
                self.task != 'ts_forecasting']).either(
            left_function=self._transformation_for_ts_forecasting,
            right_function=self._transformation_for_other_task)

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
        if self.input_data.target is not None and isinstance(
                self.input_data.target.ravel()[0],
                np.str_) and self.task == 'regression':
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
        if not self.data_convertor.is_torchvision_dataset:
            self._check_input_data_features()
            self._check_input_data_target()
        self.input_data.supplementary_data.is_auto_preprocessed = True
        return self.input_data

    def get_target_encoder(self):
        return self.label_encoder
