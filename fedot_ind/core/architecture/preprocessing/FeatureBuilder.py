from abc import ABC
import pandas as pd

class FeatureBuilderSelector:
    """Selects the appropriate feature builder based on the operation name.

    Args:
        operation_name (str): Name of the operation.
        feature_generator (callable): Function that generates features.

    """
    def __init__(self, operation_name: str, feature_generator: callable):
        self.operation_name = operation_name
        self.feature_generator = feature_generator

    def select_transformation(self):
        if self.operation_name.startswith('window'):
            return WindowBuilder(self.feature_generator).add_transformation
        elif self.operation_name.startswith('random'):
            return RandomBuilder(self.feature_generator).add_transformation
        else:
            return SteadyBuilder(self.feature_generator).add_transformation


class BuilderBase(ABC):
    """Abstract class for feature builders.

    """
    def __init__(self, feature_generator: callable):
        self.feature_generator = feature_generator

    def add_transformation(self, feature_array: pd.DataFrame, window_size: None):
        pass


class WindowBuilder(BuilderBase):
    def add_transformation(self, feature_array: pd.DataFrame, window_size: None):
        if window_size is None:
            self.window_size = round(feature_array.shape[1] / 10)
        subseq_generator = range(0, feature_array.shape[1], self.window_size)
        slice_ts = [feature_array.iloc[:, i:i + self.window_size] for i in subseq_generator]
        slice_ts = list(filter(lambda x: x.shape[1] > 1, slice_ts))
        feature_list = map(lambda x: self.feature_generator(x), slice_ts)
        X = pd.concat(feature_list)
        del feature_list
        return X


class RandomBuilder(BuilderBase):
    def add_transformation(self, feature_generator: callable, **kwargs):
        pass


class SteadyBuilder(BuilderBase):
    def add_transformation(self, feature_array: pd.DataFrame, window_size: None):
        X = pd.concat(self.feature_generator(feature_array))
        return X
