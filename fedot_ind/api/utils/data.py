import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from sklearn.preprocessing import LabelEncoder


def check_multivariate_data(data: pd.DataFrame) -> bool:
    if isinstance(data.iloc[0, 0], pd.Series):
        return True
    else:
        return False


def init_input_data(X: pd.DataFrame, y: np.ndarray, task: str = 'classification') -> InputData:
    is_multivariate_data = check_multivariate_data(X)
    task_dict = {'classification': Task(TaskTypesEnum.classification),
                 'regression': Task(TaskTypesEnum.regression)}
    features = X.values

    if type((y)[0]) is np.str_ and task == 'classification':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    elif type((y)[0]) is np.str_ and task == 'regression':
        y = y.astype(float)

    if is_multivariate_data:
        input_data = InputData(idx=np.arange(len(X)),
                               features=np.array(features.tolist()).astype(np.float),
                               target=y.reshape(-1, 1),
                               task=task_dict[task],
                               data_type=DataTypesEnum.image)
    else:
        input_data = InputData(idx=np.arange(len(X)),
                               features=X.values,
                               target=np.ravel(y).reshape(-1, 1),
                               task=task_dict[task],
                               data_type=DataTypesEnum.table)

    if task == 'regression':
        input_data.target = input_data.target.squeeze()
    elif task == 'classification':
        input_data.target[input_data.target == -1] = 0
    input_data.features = np.where(np.isnan(input_data.features), 0, input_data.features)
    input_data.features = np.where(np.isinf(input_data.features), 0, input_data.features)
    return input_data
