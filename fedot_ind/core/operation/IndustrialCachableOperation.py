import os
from typing import Optional

import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.core.operation.caching import DataCacher


class IndustrialCachableOperationImplementation(DataOperationImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

        cache_folder = os.path.join(PROJECT_PATH, 'cache')
        os.makedirs(cache_folder, exist_ok=True)
        self.cacher = DataCacher(data_type_prefix=f'Features of basis',
                                 cache_folder=cache_folder)

        self.data_type = DataTypesEnum.image

    def fit(self, data):
        """Decomposes the given data on the chosen basis.

        Returns:
            np.array: The decomposition of the given data.
        """
        pass

    def try_load_from_cache(self, hashed_info: str) -> np.array:
        predict = self.cacher.load_data_from_cache(hashed_info=hashed_info)
        return predict

    def transform(self, input_data: InputData, use_cache: bool = False) -> OutputData:
        """Method firstly tries to load result from cache. If unsuccessful, it starts to generate features
        """
        if use_cache:
            class_params = {k: v for k, v in self.__dict__.items() if k not in ['cacher',
                                                                                'data_type',
                                                                                'params',
                                                                                'n_processes',
                                                                                'logging_params',
                                                                                'logger',
                                                                                'relevant_features']}

            hashed_info = self.cacher.hash_info(data=input_data.features,
                                                operation_info=class_params.__repr__())
            try:
                predict = self.try_load_from_cache(hashed_info)
            except FileNotFoundError:
                predict = self._transform(input_data)
                self.cacher.cache_data(hashed_info, predict)

            predict = self._convert_to_output(input_data, predict, data_type=self.data_type)
            return predict
        else:
            predict = self._transform(input_data)
            predict = self._convert_to_output(input_data, predict, data_type=self.data_type)
            return predict

    def _transform(self, input_data):
        pass
