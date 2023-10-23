import numpy as np
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot_ind.core.architecture.preprocessing.data_convertor import DataConverter


def fedot_data_type(func):
    def decorated_func(self, *args):
        if type(args[0]) is not InputData:
            args[0] = DataConverter(data=args[0])
        features = args[0].features

        try:
            input_data_squeezed = np.squeeze(features, 3)
        except Exception:
            input_data_squeezed = np.squeeze(features)

        return func(self, input_data_squeezed)

    return decorated_func


def convert_to_3d_torch_array(func):
    def decorated_func(self, *args):
        data = DataConverter(data=args[0]).convert_to_torch_format()
        return func(self, data)

    return decorated_func


def remove_1_dim_axis(func):
    def decorated_func(self, *args):
        time_series = np.nan_to_num(args[0])
        if any(time_series.shape) == 1:
            time_series = DataConverter(data=time_series).convert_to_1d_array()
        return func(self, time_series)

    return decorated_func


def convert_to_input_data(func):
    def decorated_func(*args, **kwargs):
        features, names = func(*args, **kwargs)
        ts_data = InputData(idx=np.arange(len(features)),
                            features=features,
                            target='no_target',
                            task='no_task',
                            data_type=DataTypesEnum.table,
                            supplementary_data={'feature_name': names})
        return ts_data

    return decorated_func
