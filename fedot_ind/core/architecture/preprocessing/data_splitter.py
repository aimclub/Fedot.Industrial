import numpy as np
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from sklearn.model_selection import train_test_split


class DataSplitter:
    """Класс для разделения данных на обучающую и тестовую выборки"""

    def __init__(self, test_size: float = 0.2, random_state: int = 42,
                 temporal_split: bool = True):
        self.test_size = test_size
        self.random_state = random_state
        self.temporal_split = temporal_split

    def split(self, input_data: InputData) -> tuple:
        """Разделение данных на train/test"""
        X, y = input_data.features, input_data.target
        if self.temporal_split:
            # Временное разделение (для временных рядов)
            split_idx = int(len(X) * (1 - self.test_size))
            X_train = X[:split_idx]
            X_test = X[split_idx:]

            if y is not None:
                y_train = y[:split_idx]
                y_test = y[split_idx:]
            else:
                y_train, y_test = None, None
        else:
            # Случайное разделение
            if y is not None:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_size, random_state=self.random_state,
                    stratify=y if len(np.unique(y)) > 1 else None
                )
            else:
                X_train, X_test = train_test_split(
                    X, test_size=self.test_size, random_state=self.random_state
                )
                y_train, y_test = None, None
        input_data_train = InputData(idx=np.arange(len(X_train)),
                                     features=X_train,
                                     target=y_train,
                                     task=Task(TaskTypesEnum.classification),
                                     data_type=DataTypesEnum.table)
        input_data_test = InputData(idx=np.arange(len(X_test)),
                                    features=X_test,
                                    target=y_test,
                                    task=Task(TaskTypesEnum.classification),
                                    data_type=DataTypesEnum.table)
        return input_data_train, input_data_test
