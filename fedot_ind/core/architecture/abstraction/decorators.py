import numpy as np
import torch
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from torch.utils.data import DataLoader

from fedot_ind.core.architecture.preprocessing.data_convertor import DataConverter, TensorConverter, \
    CustomDatasetCLF, CustomDatasetTS


def fedot_data_type(func):
    def decorated_func(self, *args):
        if type(args[0]) is not InputData:
            args[0] = DataConverter(data=args[0])
        features = args[0].features

        if len(features.shape) < 4:
            try:
                input_data_squeezed = np.squeeze(features, 3)
            except Exception:
                input_data_squeezed = np.squeeze(features)
        else:
            input_data_squeezed = features

        return func(self, input_data_squeezed)

    return decorated_func


def convert_to_3d_torch_array(func):
    def decorated_func(self, *args):
        init_data = args[0]
        data = DataConverter(data=init_data).convert_to_torch_format()
        if type(init_data) is InputData:
            init_data.features = data
        else:
            init_data = data
        return func(self, init_data)

    return decorated_func


def convert_inputdata_to_torch_dataset(func):
    def decorated_func(self, *args):
        ts = args[0]
        return func(self, CustomDatasetCLF(ts))

    return decorated_func


def convert_inputdata_to_torch_time_series_dataset(func):
    def decorated_func(self, *args):
        ts = args[0]
        return func(self, CustomDatasetTS(ts))

    return decorated_func


def convert_to_torch_tensor(func):
    def decorated_func(self, *args):
        data = TensorConverter(data=args[0]).convert_to_tensor(data=args[0])
        return func(self, data)

    return decorated_func


def remove_1_dim_axis(func):
    def decorated_func(self, *args):
        time_series = np.nan_to_num(args[0])
        if any([dim == 1 for dim in time_series.shape]):
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
