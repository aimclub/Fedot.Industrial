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

    def extract_stats_features_torch(self, ts: torch.Tensor, axis: int) -> InputData:
        """Method for extracting statistical features for data and its windows and 
        concatenating results. It extracts only base features for windows.
        """
        global_features = self.get_statistical_features_torch(ts, add_global_features=self.add_global_features, axis=axis)
        if ts.squeeze().ndim == 1:
            global_features = torch.Tensor(global_features)
        else:
            global_features = torch.stack(global_features, dim=0).T

        if self.window_size == 0:
            window_stat_features = self.get_statistical_features_torch(ts, 
                                                                       axis=axis) 
        else:
            window_stat_features = self.apply_window_for_stat_feature_torch(ts_data=ts, 
                                                     feature_generator=self.get_statistical_features_torch,
                                                     window_size=self.window_size)

        if self.add_global_features:
            window_stat_features = window_stat_features
            if window_stat_features.ndim > 2:
                window_stat_features = window_stat_features.reshape(window_stat_features.shape[0],
                                                                    window_stat_features.shape[-1] * 
                                                            window_stat_features.shape[-2]).squeeze()
            else:
                window_stat_features = window_stat_features.reshape(window_stat_features.shape[-1] * 
                                                            window_stat_features.shape[-2]).squeeze()
            return torch.cat([global_features, window_stat_features], dim=-1)
        else:
            return window_stat_features

    @dask.delayed 
    def generate_features_from_ts(self, ts: torch.Tensor) -> torch.Tensor:
        """
        Method for Tensor or batch of Tensors. Use only last axis.

        Args:
            ts: Tensor with dimension 1 or 2
        Returns:
            torch.Tensor: statistical features
        """
        if ts.ndim == 1:
            ts = ts.unsqueeze(0)
        features = self.extract_stats_features_torch(ts, axis=-1)
        return features
    
    def generate_features_from_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Method for a tensor with one dimention.
        """
        statistical_representation = self.get_statistical_features_torch(tensor,
                                                                   add_global_features=self.add_global_features,
                                                                   axis=2)
        l = [x for x in statistical_representation if x is not None]
        return l
