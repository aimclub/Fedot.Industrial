import torch
from typing import Optional
import dask

from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.models.base_extractor import BaseExtractor


class TorchQuantileExtractor(BaseExtractor):
    """Class responsible for statistical feature generator experiment.

    Attributes:
        window_size (int): size of window
        stride (int): stride for window
        var_threshold (float): threshold for variance
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.window_size = params.get('window_size', 0)
        self.stride = params.get('stride', 1)
        self.add_global_features = params.get('add_global_features', True)
        self.logging_params.update({'Wsize': self.window_size,
                                    'Stride': self.stride})

    def __repr__(self):
        return 'Statistical Class for TS representation'

    def _concatenate_global_and_local_feature(self,
                                              global_features: torch.Tensor,
                                              window_stat_features: torch.Tensor) -> torch.Tensor:
        window_stat_features = window_stat_features.squeeze()
        if window_stat_features.shape[0] > 2:
            window_stat_features = window_stat_features.flatten()
        concatenated_features = torch.cat([global_features, window_stat_features], dim=0)
        concatenated_features = torch.nan_to_num(concatenated_features)
        return concatenated_features

    def extract_stats_features_torch(self, ts: torch.Tensor, axis: int) -> InputData:
        global_features = self.get_statistical_features_torch(ts, add_global_features=self.add_global_features, axis=axis)
        global_features = torch.Tensor(global_features)
        window_stat_features = self.get_statistical_features_torch(ts, axis=axis) if self.window_size == 0 else \
            self.apply_window_for_stat_feature_torch(ts_data=ts, feature_generator=self.get_statistical_features_torch,
                                               window_size=self.window_size)
        window_stat_features = torch.Tensor(window_stat_features)
        return self._concatenate_global_and_local_feature(
            global_features, window_stat_features) if self.add_global_features else window_stat_features

    @dask.delayed
    def generate_features_from_ts(self, ts: torch.Tensor):
        if ts.ndim == 1:
            ts = ts.unsqueeze(0)
        features = [self.extract_stats_features_torch(channel, axis=-1) for channel in ts]
        statistical_representation = torch.stack(features, dim=0)
        return statistical_representation
    
    def generate_features_from_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Method for an tensor with one dimention.
        """
        statistical_representation = self.get_statistical_features_torch(tensor,
                                                                   add_global_features=self.add_global_features,
                                                                   axis=2)
        l = [x for x in statistical_representation if x is not None]
        return l
