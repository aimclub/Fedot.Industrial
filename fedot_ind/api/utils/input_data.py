import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def init_input_data(X: pd.DataFrame, y: np.ndarray, task: str = 'classification') -> InputData:
    """Method for initialization of InputData object from pandas DataFrame and numpy array with target values.

    Args:
        X: pandas DataFrame with features
        y: numpy array with target values

    Returns:
        InputData object convenient for FEDOT framework

    """
    is_multivariate_data = True if isinstance(X.iloc[0, 0], pd.Series) else False
    if is_multivariate_data:
        input_data = InputData(idx=np.arange(len(X)),
                               features=np.array(X.values.tolist()),
                               target=y.reshape(-1, 1),
                               # task=Task(TaskTypesEnum.classification),
                               task=Task(TaskTypesEnum(task)),
                               data_type=DataTypesEnum.image)
    else:
        input_data = InputData(idx=np.arange(len(X)),
                               features=X.values,
                               target=np.ravel(y).reshape(-1, 1),
                               # task=Task(TaskTypesEnum.classification),
                               task=Task(TaskTypesEnum(task)),
                               data_type=DataTypesEnum.table)

    return input_data
