import pandas as pd


class FeatureGeneratorComposer:
    def __init__(self):
        self.dict = {}

    def __getitem__(self, item):
        return self.dict[item]

    def add_operation(self, operation_name: str,
                      operation_functionality: object):
        self.dict[operation_name] = operation_functionality


class FeatureGeneratorBuilder:
    def __init__(self, feature_generator: callable):
        self.feature_generator = feature_generator

    def add_window_transformation(self, feature_array: pd.DataFrame, window_size: None):
        if window_size is None:
            self.window_size = round(feature_array.shape[1] / 10)
        subseq_generator = range(0, feature_array.shape[1], self.window_size)
        slice_ts = [feature_array.iloc[:, i:i + self.window_size] for i in subseq_generator]
        slice_ts = list(filter(lambda x: x.shape[1] > 1, slice_ts))
        feature_list = map(lambda x: self.feature_generator(x), slice_ts)
        # columns_name = [x + f'_on_interval: {i} - {i + self.window_size}' for x in df.columns for i in
        #                 subseq_generator]
        X = pd.concat(feature_list)
        del feature_list
        return X

    def add_steady_transformation(self, feature_array: pd.DataFrame, window_size: None):
        X = pd.concat(self.feature_generator(feature_array))
        return X

    def add_random_interval_transformation(self, feature_generator: callable):
        # subseq_generator = range(0, self.feature_array.shape[1], self.window_size)
        # slice_ts = [self.feature_array.iloc[:, i:i + self.window_size] for i in subseq_generator]
        # slice_ts = list(filter(lambda x: x.shape[1] > 1, slice_ts))
        # feature_list = map(lambda x: feature_generator(x), slice_ts)
        # # columns_name = [x + f'_on_interval: {i} - {i + self.window_size}' for x in df.columns for i in
        # #                 subseq_generator]
        # X = pd.concat(feature_list)
        # del feature_list
        # return X
        pass
