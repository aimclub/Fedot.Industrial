import logging
from typing import Union

import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from sklearn.preprocessing import LabelEncoder

from fedot_ind.api.utils.data import check_multivariate_data
from fedot_ind.core.architecture.preprocessing.data_convertor import NumpyConverter
from fedot_ind.core.architecture.settings.computational import backend_methods as np


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
                 input_data: Union[tuple, InputData],
                 task: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.input_data = input_data
        self.task = task
        self.task_dict = {'classification': Task(TaskTypesEnum.classification),
                          'regression': Task(TaskTypesEnum.regression)}

    def _init_input_data(self) -> None:
        """Initializes the `input_data` attribute based on its type.

        If a tuple (X, y) is provided, it converts it to a Fedot InputData object
        with appropriate data types and task information. If an existing InputData
        object is provided, it checks if it requires further initialization.

        Raises:
            ValueError: If the input data format is invalid.

        """

        if isinstance(self.input_data, tuple):
            X, y = self.input_data[0], self.input_data[1]
            if type(X) is not pd.DataFrame:
                X = pd.DataFrame(X)
            is_multivariate_data = check_multivariate_data(X)

            if is_multivariate_data:
                self.input_data = InputData(idx=np.arange(len(X)),
                                            features=np.array(X.values.tolist()).astype(np.float),
                                            target=y.reshape(-1, 1),
                                            task=self.task_dict[self.task],
                                            data_type=DataTypesEnum.image)
            else:
                self.input_data = InputData(idx=np.arange(len(X)),
                                            features=X.values,
                                            target=np.ravel(y).reshape(-1, 1),
                                            task=self.task_dict[self.task],
                                            data_type=DataTypesEnum.image)
        elif type(self.input_data) is InputData:
            return
        else:
            raise ValueError(f"Invalid input data format: {type(self.input_data)}")

    def _check_input_data_features(self) -> None:
        """Checks and preprocesses the features in the input data.

        - Replaces NaN and infinite values with 0.
        - Converts features to torch format using NumpyConverter.

        """

        self.input_data.features = np.where(
            np.isnan(self.input_data.features), 0, self.input_data.features)
        self.input_data.features = np.where(
            np.isinf(self.input_data.features), 0, self.input_data.features)
        self.input_data.features = NumpyConverter(
            data=self.input_data.features).convert_to_torch_format()

        if self.task == 'regression':
            self.input_data.target = self.input_data.target.squeeze()
        elif self.task == 'classification':
            self.input_data.target[self.input_data.target == -1] = 0

    def _check_input_data_target(self):
        """Checks and preprocesses the target variable in the input data.

        - Encodes labels if the task is classification.
        - Casts the target variable to float if the task is regression.

        """
        if self.input_data.target is not None and type(self.input_data.target.ravel()[0]) is np.str_ and self.task == 'regression':
            self.input_data.target = self.input_data.target.astype(float)

        if self.task == 'regression':
            self.input_data.target = self.input_data.target.squeeze()
        elif self.task == 'classification':
            self.input_data.target[self.input_data.target == -1] = 0

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
