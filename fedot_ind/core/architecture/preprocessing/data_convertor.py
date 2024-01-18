from functools import partial

import pandas as pd
import torch
import torch.nn as nn
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from pymonad.list import ListMonad
from sklearn.preprocessing import LabelEncoder

from examples.example_utils import check_multivariate_data
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.architecture.settings.computational import default_device
from fedot_ind.core.repository.constanst_repository import MATRIX, MULTI_ARRAY


class CustomDatasetTS:
    def __init__(self, ts):
        self.x = torch.from_numpy(DataConverter(
            data=ts.features).convert_to_torch_format()).float()
        self.y = torch.from_numpy(DataConverter(
            data=ts.target).convert_to_torch_format()).float()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class CustomDatasetCLF:
    def __init__(self, ts):
        self.x = torch.from_numpy(ts.features).to(default_device()).float()
        if ts.task.task_type == 'classification':
            label_1 = max(ts.class_labels)
            label_0 = min(ts.class_labels)
            self.classes = ts.num_classes
            if self.classes == 2 and label_1 != 1:
                ts.target[ts.target == label_0] = 0
                ts.target[ts.target == label_1] = 1
            elif self.classes == 2 and label_0 != 0:
                ts.target[ts.target == label_0] = 0
                ts.target[ts.target == label_1] = 1
            elif self.classes > 2 and label_0 == 1:
                ts.target = ts.target - 1
            if type(min(ts.target)[0]) is np.str_:
                self.label_encoder = LabelEncoder()
                ts.target = self.label_encoder.fit_transform(ts.target)
            else:
                self.label_encoder = None

            try:
                self.y = torch.nn.functional.one_hot(torch.from_numpy(ts.target).long(),
                                                     num_classes=self.classes).to(default_device()).squeeze(1)
            except Exception:
                self.y = torch.nn.functional.one_hot(torch.from_numpy(
                    ts.target).long()).to(default_device()).squeeze(1)
                self.classes = self.y.shape[1]
        else:
            self.y = torch.from_numpy(ts.target).to(default_device()).float()
            self.classes = 1
            self.label_encoder = None

        self.n_samples = ts.features.shape[0]
        self.supplementary_data = ts.supplementary_data

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class FedotConverter:
    def __init__(self, data):
        self.input_data = self.convert_to_input_data(data)

    def convert_to_input_data(self, data):
        if isinstance(data, InputData):
            return data
        elif isinstance(data[0], (np.ndarray, pd.DataFrame)):
            return self.__init_input_data(features=data[0], target=data[1])
        else:
            try:
                return torch.tensor(data)
            except:
                print(f"Can't convert {type(data)} to InputData", Warning)

    def __init_input_data(self, features: pd.DataFrame,
                          target: np.ndarray,
                          task: str = 'classification') -> InputData:
        if type(features) is np.ndarray:
            features = pd.DataFrame(features)
        is_multivariate_data = check_multivariate_data(features)
        task_dict = {'classification': Task(TaskTypesEnum.classification),
                     'regression': Task(TaskTypesEnum.regression)}
        if is_multivariate_data:
            input_data = InputData(idx=np.arange(len(features)),
                                   features=np.array(
                                       features.values.tolist()).astype(np.float),
                                   target=target.astype(
                                       np.float).reshape(-1, 1),
                                   task=task_dict[task],
                                   data_type=MULTI_ARRAY)
        else:
            input_data = InputData(idx=np.arange(len(features)),
                                   features=features.values,
                                   target=np.ravel(target).reshape(-1, 1),
                                   task=task_dict[task],
                                   data_type=MATRIX)
        return input_data


class TensorConverter:
    def __init__(self, data):
        self.tensor_data = self.convert_to_tensor(data)

    def convert_to_tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, pd.DataFrame):
            return torch.from_numpy(data.values)
        elif isinstance(data, InputData):
            return torch.from_numpy(data.features)
        else:
            print(f"Can't convert {type(data)} to torch.Tensor", Warning)

    def convert_to_1d_tensor(self):
        if self.tensor_data.ndim == 1:
            return self.tensor_data
        elif self.tensor_data.ndim == 3:
            return self.tensor_data[0, 0]
        if self.tensor_data.ndim == 2:
            return self.tensor_data[0]
        assert False, f'Please, review input dimensions {self.tensor_data.ndim}'

    def convert_to_2d_tensor(self):
        if self.tensor_data.ndim == 2:
            return self.tensor_data
        elif self.tensor_data.ndim == 1:
            return self.tensor_data[None]
        elif self.tensor_data.ndim == 3:
            return self.tensor_data[0]
        assert False, f'Please, review input dimensions {self.tensor_data.ndim}'

    def convert_to_3d_tensor(self):
        if self.tensor_data.ndim == 3:
            return self.tensor_data
        elif self.tensor_data.ndim == 1:
            return self.tensor_data[None, None]
        elif self.tensor_data.ndim == 2:
            return self.tensor_data[:, None]
        assert False, f'Please, review input dimensions {self.tensor_data.ndim}'


class NumpyConverter:
    def __init__(self, data):
        self.numpy_data = self.convert_to_array(data)

    def convert_to_array(self, data):
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.detach().numpy()
        elif isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, InputData):
            return data.features
        elif isinstance(data, CustomDatasetTS):
            return data.x
        elif isinstance(data, CustomDatasetCLF):
            return data.x
        else:
            try:
                return np.asarray(data)
            except:
                print(f"Can't convert {type(data)} to np.array", Warning)

    def convert_to_1d_array(self):
        if self.numpy_data.ndim == 1:
            return self.numpy_data
        elif self.numpy_data.ndim > 2:
            return np.squeeze(self.numpy_data)
        elif self.numpy_data.ndim == 2:
            return self.numpy_data.flatten()
        assert False, print(
            f'Please, review input dimensions {self.numpy_data.ndim}')

    def convert_to_2d_array(self):
        if self.numpy_data.ndim == 2:
            return self.numpy_data
        elif self.numpy_data.ndim == 1:
            return self.numpy_data.reshape(1, -1)
        elif self.numpy_data.ndim == 3:
            return self.numpy_data[0]
        assert False, print(
            f'Please, review input dimensions {self.numpy_data.ndim}')

    def convert_to_3d_array(self):
        if self.numpy_data.ndim == 3:
            return self.numpy_data
        elif self.numpy_data.ndim == 1:
            return self.numpy_data[None, None]
        elif self.numpy_data.ndim == 2:
            return self.numpy_data[:, None]
        assert False, print(
            f'Please, review input dimensions {self.numpy_data.ndim}')

    def convert_to_torch_format(self):
        if self.numpy_data.ndim == 3:
            return self.numpy_data
        elif self.numpy_data.ndim == 1:
            return self.numpy_data.reshape(self.numpy_data.shape[0],
                                           1,
                                           1)
        elif self.numpy_data.ndim == 2:
            return self.numpy_data.reshape(self.numpy_data.shape[0],
                                           1,
                                           self.numpy_data.shape[1])
        elif self.numpy_data.ndim > 3:
            return self.numpy_data.squeeze()
        assert False, print(
            f'Please, review input dimensions {self.numpy_data.ndim}')


class DataConverter(TensorConverter, NumpyConverter):
    def __init__(self, data):
        super().__init__(data)
        self.data = data
        self.numpy_data = self.convert_to_array(data)

    @property
    def is_nparray(self):
        return isinstance(self.data, np.ndarray)

    @property
    def is_tensor(self):
        return isinstance(self.data, torch.Tensor)

    @property
    def is_zarr(self):
        return hasattr(self.data, 'oindex')

    @property
    def is_dask(self):
        return hasattr(self.data, 'compute')

    @property
    def is_memmap(self):
        return isinstance(self.data, np.memmap)

    @property
    def is_slice(self):
        return isinstance(self.data, slice)

    @property
    def is_tuple(self):
        return isinstance(self.data, tuple)

    @property
    def is_none(self):
        return self.data is None

    @property
    def is_fedot_data(self):
        return isinstance(self.data, InputData)

    @property
    def is_exist(self):
        return self.data is not None

    def convert_to_data_type(self):
        if isinstance(self.data, torch.Tensor):
            self.data = self.data.to(dtype=torch.Tensor)
        elif isinstance(self.data, np.ndarray):
            self.data = self.data.astype(np.ndarray)

    def convert_to_list(self):
        if isinstance(self.data, list):
            return self.data
        elif isinstance(self.data, (np.ndarray, torch.Tensor)):
            return self.data.tolist()
        else:
            try:
                return list(self.data)
            except:
                print(f'passed object needs to be of type L, list, np.ndarray or torch.Tensor but is {type(self.data)}',
                      Warning)

    def convert_data_to_1d(self):
        if self.data.ndim == 1:
            return self.data
        if isinstance(self.data, np.ndarray):
            return self.convert_to_1d_array()
        if isinstance(self.data, torch.Tensor):
            return self.convert_to_1d_tensor()

    def convert_data_to_2d(self):
        if self.data.ndim == 2:
            return self.data
        if isinstance(self.data, np.ndarray):
            return self.convert_to_2d_array()
        if isinstance(self.data, torch.Tensor):
            return self.convert_to_2d_tensor()

    def convert_data_to_3d(self):
        if self.data.ndim == 3:
            return self.data
        if isinstance(self.data, (np.ndarray, pd.self.dataFrame)):
            return self.convert_to_3d_array()
        if isinstance(self.data, torch.Tensor):
            return self.convert_to_3d_tensor()

    def convert_to_monad_data(self):
        if self.is_fedot_data:
            features = np.array(ListMonad(*self.data.features.tolist()).value)
        else:
            features = np.array(ListMonad(*self.data.tolist()).value)

        if len(features.shape) == 2 and features.shape[1] == 1:
            features = features.reshape(1, -1)
        elif len(features.shape) == 3 and features.shape[1] == 1:
            features = features.squeeze()
        return features

    def convert_to_eigen_basis(self):
        if self.is_fedot_data:
            features = self.data.features
        else:
            features = np.array(ListMonad(*self.data.values.tolist()).value)
            features = np.array([series[~np.isnan(series)]
                                for series in features])
        return features


class NeuralNetworkConverter:
    def __init__(self, layer):
        self.layer = layer

    @property
    def is_layer(self, *args):
        def _is_layer(cond=args):
            return isinstance(self.layer, cond)

        return partial(_is_layer, cond=args)

    @property
    def is_linear(self):
        return isinstance(self.layer, nn.Linear)

    @property
    def is_batch_norm(self):
        types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
        return isinstance(self.layer, types)

    @property
    def is_convolutional_linear(self):
        types = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
        return isinstance(self.layer, types)

    @property
    def is_affine(self):
        return self.has_bias or self.has_weight

    @property
    def is_convolutional(self):
        types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
        return isinstance(self.layer, types)

    @property
    def has_bias(self):
        return hasattr(self.layer, 'bias') and self.layer.bias is not None

    @property
    def has_weight(self):
        return hasattr(self.layer, 'weight')

    @property
    def has_weight_or_bias(self):
        return any((self.has_weight, self.has_bias))
