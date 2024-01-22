import logging

import pandas as pd

from fedot_ind.api.utils.data import check_multivariate_data
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from sklearn.preprocessing import LabelEncoder
from fedot.core.repository.tasks import Task, TaskTypesEnum

from fedot_ind.core.architecture.preprocessing.data_convertor import NumpyConverter


class DataCheck:
    def __init__(self,
                 input_data,
                 task):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.input_data = input_data
        self.task = task
        self.task_dict = {'classification': Task(TaskTypesEnum.classification),
                          'regression': Task(TaskTypesEnum.regression)}

    def _init_input_data(self):

        if type(self.input_data) is tuple:
            X, y = self.input_data[0], self.input_data[1]
            if type(X) is not pd.DataFrame:
                X = pd.DataFrame(X)
            is_multivariate_data = check_multivariate_data(X)

            if is_multivariate_data:
                self.input_data = InputData(idx=np.arange(len(X)),
                                            features=np.array(
                                                X.values.tolist()).astype(np.float),
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

    def _check_input_data_features(self):
        self.input_data.features = np.where(
            np.isnan(self.input_data.features), 0, self.input_data.features)
        self.input_data.features = np.where(
            np.isinf(self.input_data.features), 0, self.input_data.features)
        self.input_data.features = NumpyConverter(
            data=self.input_data.features).convert_to_torch_format()

    def _check_input_data_target(self):
        if type(self.input_data.target[0][0]) is np.str_ and self.task == 'classification':
            label_encoder = LabelEncoder()
            self.input_data.target = label_encoder.fit_transform(
                self.input_data.target)
        elif type(self.input_data.target[0][0]) is np.str_ and self.task == 'regression':
            self.input_data.target = self.input_data.target.astype(float)

        if self.task == 'regression':
            self.input_data.target = self.input_data.target.squeeze()
        elif self.task == 'classification':
            self.input_data.target[self.input_data.target == -1] = 0

    def check_input_data(self):
        self._init_input_data()
        self._check_input_data_features()
        self._check_input_data_target()
        return self.input_data
