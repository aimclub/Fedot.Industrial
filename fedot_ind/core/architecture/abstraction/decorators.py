import numpy as np
from fedot.core.data.data import InputData

from fedot_ind.core.architecture.preprocessing.data_convertor import FedotConverter, NumpyConverter


def fedot_data_type(func):
    def decorated_func(self, *args):
        if type(args[0]) is not InputData:
            args[0] = FedotConverter(data=args[0])
        features = args[0].features

        try:
            input_data_squeezed = np.squeeze(features, 3)
        except Exception:
            input_data_squeezed = np.squeeze(features)

        return func(self, input_data_squeezed)

    return decorated_func


def convert_to_3d_torch_array(func):
    def decorated_func(self, *args):
        data = NumpyConverter(data=args[0]).convert_to_torch_format()
        return func(self, data)

    return decorated_func
