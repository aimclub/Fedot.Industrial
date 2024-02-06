from typing import Optional

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from sklearn.preprocessing import LabelEncoder


def check_multivariate_data(data: pd.DataFrame) -> tuple:
    """
    Checks if the provided pandas DataFrame contains multivariate data.

    Args:
        data (pd.DataFrame): The DataFrame to be analyzed.

    Returns:
        bool: True if the DataFrame contains multivariate data (nested columns), False otherwise.
    """
    if type(data) is not pd.DataFrame:
        return len(data.shape) > 2, data
    else:
        return isinstance(data.iloc[0, 0], pd.Series), data.values


def init_input_data(X: pd.DataFrame,
                    y: Optional[np.ndarray],
                    task: str = 'classification') -> InputData:
    """
    Initializes a Fedot InputData object from input features and target.

    Args:
        X: The DataFrame containing features.
        y: The NumPy array containing target values.
        task: The machine learning task type ("classification" or "regression"). Defaults to "classification".

    Returns:
        InputData: The initialized Fedot InputData object.

    """

    is_multivariate_data, features = check_multivariate_data(X)
    task_dict = {'classification': Task(TaskTypesEnum.classification),
                 'regression': Task(TaskTypesEnum.regression)}

    if y is not None and type(y[0]) is np.str_ and task == 'classification':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    elif y is not None and type(y[0]) is np.str_ and task == 'regression':
        y = y.astype(float)

    data_type = DataTypesEnum.image if is_multivariate_data else DataTypesEnum.table
    input_data = InputData(idx=np.arange(len(X)),
                           features=np.array(features.tolist()).astype(float),
                           target=y.reshape(-1, 1) if y is not None else y,
                           task=task_dict[task],
                           data_type=data_type)

    if input_data.target is not None:
        if task == 'regression':
            input_data.target = input_data.target.squeeze()
        elif task == 'classification':
            input_data.target[input_data.target == -1] = 0

    # Replace NaN and infinite values with 0 in features
    input_data.features = np.where(
        np.isnan(input_data.features), 0, input_data.features)
    input_data.features = np.where(
        np.isinf(input_data.features), 0, input_data.features)

    return input_data
