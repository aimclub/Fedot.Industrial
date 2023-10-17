from functools import partial
import numpy as np
import pandas as pd

import torch
import torch.nn as nn


class TensorConverter:
    def __init__(self, data):
        self.tensor_data = self.convert_to_tensor(data)

    def convert_to_tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, pd.self.dataFrame):
            return torch.from_numpy(data.values)
        else:
            try:
                return torch.tensor(data)
            except:
                print(f"Can't convert {type(data)} to torch.Tensor", Warning)

    def convert_to_1d_tensor(self):
        if self.tensor_data.ndim == 1:
            return self.tensor_data
        elif self.tensor_data.ndim == 3:
            return self.tensor_data[0, 0]
        if self.tensor_data.ndim == 2: return self.tensor_data[0]
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
            return data.cpu().numpy()
        elif isinstance(data, pd.self.dataFrame):
            return data.values
        else:
            try:
                return np.asarray(data)
            except:
                print(f"Can't convert {type(data)} to np.array", Warning)

    def convert_to_1d_array(self):
        if self.numpy_data.ndim == 1:
            return self.numpy_data
        elif self.numpy_data.ndim == 3:
            self.numpy_data = self.numpy_data[0, 0]
        elif self.numpy_data.ndim == 2:
            self.numpy_data = self.numpy_data[0]
        assert False, print(f'Please, review input dimensions {self.numpy_data.ndim}')

    def convert_to_2d_array(self):
        if self.numpy_data.ndim == 2:
            return self.numpy_data
        elif self.numpy_data.ndim == 1:
            return self.numpy_data[None]
        elif self.numpy_data.ndim == 3:
            return self.numpy_data[0]
        assert False, print(f'Please, review input dimensions {self.numpy_data.ndim}')

    def convert_to_3d_array(self):
        if self.numpy_data.ndim == 3:
            return self.numpy_data
        elif self.numpy_data.ndim == 1:
            return self.numpy_data[None, None]
        elif self.numpy_data.ndim == 2:
            return self.numpy_data[:, None]
        assert False, print(f'Please, review input dimensions {self.numpy_data.ndim}')


class DataConverter(TensorConverter, NumpyConverter):
    def __init__(self, data):
        super().__init__(data)
        self.data = data

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
        if self.data.ndim == 1: return self.data
        if isinstance(self.data, np.ndarray): return self.convert_to_1d_array()
        if isinstance(self.data, torch.Tensor): return self.convert_to_1d_tensor()

    def convert_data_to_2d(self):
        if self.data.ndim == 2: return self.data
        if isinstance(self.data, np.ndarray): return self.convert_to_2d_array()
        if isinstance(self.data, torch.Tensor): return self.convert_to_2d_tensor()

    def convert_data_to_3d(self):
        if self.data.ndim == 3: return self.data
        if isinstance(self.data, (np.ndarray, pd.self.dataFrame)): return self.convert_to_3d_array()
        if isinstance(self.data, torch.Tensor): return self.convert_to_3d_tensor()


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
        return (hasattr(self.layer, 'bias') and self.layer.bias is not None)

    @property
    def has_weight(self):
        return (hasattr(self.layer, 'weight'))

    @property
    def has_weight_or_bias(self):
        return any((self.has_weight, self.has_bias))
