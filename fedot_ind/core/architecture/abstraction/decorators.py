import numpy as np
import torch
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot_ind.core.architecture.preprocessing.data_convertor import DataConverter, TensorConverter


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

        class CustomDataset:
            def __init__(self, ts):
                self.x = torch.from_numpy(ts.features).float()
                label_1 = max(ts.class_labels)
                label_0 = min(ts.class_labels)
                classes = ts.num_classes
                if classes == 2 and label_1 != 1:
                    ts.target[ts.target == label_0] = 0
                    ts.target[ts.target == label_1] = 1
                elif classes == 2 and label_0 != 0:
                    ts.target[ts.target == label_0] = 0
                    ts.target[ts.target == label_1] = 1
                elif classes > 2 and label_0 == 1:
                    ts.target = ts.target - 1

                self.y = torch.nn.functional.one_hot(torch.from_numpy(ts.target).long(),
                                                     num_classes=classes).squeeze(1)
                self.n_samples = ts.features.shape[0]

            def __getitem__(self, index):
                return self.x[index], self.y[index]

            def __len__(self):
                return self.n_samples

        return func(self, CustomDataset(ts))

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
