from typing import Optional

import torch
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.models.base_extractor import BaseExtractor


class TorchQuantileExtractor(BaseExtractor):
    """
    A PyTorch-based feature extractor for computing statistical features from time series data.

    This class extracts statistical features (such as mean, variance, quantiles, etc.) from time series,
    both globally and within sliding windows. It supports batch processing and GPU acceleration.

    Attributes:
        window_size (int): The size of the sliding window for local feature extraction. Defaults to 0.
        stride (int): The stride for the sliding window. Defaults to 1.
        add_global_features (bool): If True, global statistical features are concatenated with window-based features.
                                    Defaults to True.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.window_size = params.get('window_size', 0)
        self.stride = params.get('stride', 1)
        self.add_global_features = params.get('add_global_features', True)
        self.logging_params.update({'Wsize': self.window_size,
                                    'Stride': self.stride})

    @staticmethod
    def _as_feature_matrix(features: list, source: torch.Tensor) -> torch.Tensor:
        n_samples = source.shape[0] if source.ndim > 1 else 1
        matrices = []
        for feature in features:
            tensor = torch.as_tensor(feature, device=source.device, dtype=torch.float32)
            if tensor.ndim == 0:
                tensor = tensor.reshape(1, 1)
            elif tensor.shape[0] == n_samples:
                tensor = tensor.reshape(n_samples, -1)
            else:
                tensor = tensor.reshape(1, -1)
            matrices.append(tensor)
        return torch.cat(matrices, dim=-1)

    def extract_stats_features_torch(self, ts: torch.Tensor, axis: int) -> InputData:
        """
        This method computes global statistical features and window-based features,
        then concatenates them if `add_global_features` is True.

        Args:
            ts (torch.Tensor): Input time series tensor.
            axis (int): Axis along which to compute features.

        Returns:
            torch.Tensor: Concatenated statistical features.
                If `add_global_features` is True, returns global + window features.
                Otherwise, returns only window features.
        """
        global_features = self._as_feature_matrix(
            self.get_statistical_features_torch(
                ts,
                add_global_features=self.add_global_features,
                axis=axis,
            ),
            ts,
        )
        used_window_fallback = False
        if self.window_size == 0 or ts.shape[axis] <= 5:
            window_stat_features = self._as_feature_matrix(
                self.get_statistical_features_torch(ts, axis=axis),
                ts,
            )
            used_window_fallback = True
        else:
            window_stat_features = self.apply_window_for_stat_feature_torch(
                ts_data=ts, feature_generator=self.get_statistical_features_torch, window_size=self.window_size)
        if self.add_global_features:
            if self.window_size != 0 and not used_window_fallback:
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

    def generate_features_from_ts(self, ts: torch.Tensor) -> torch.Tensor:
        """
        Generate statistical features from a single time series or a batch of time series.

        Args:
            ts (torch.Tensor): Input tensor with dimension 1, 2, or 3.

        Returns:
            torch.Tensor: Extracted statistical features as a CPU tensor.
        """
        if ts.ndim == 1:
            ts = ts.unsqueeze(0)
        if ts.ndim > 2:
            self.is_multichanel = True
        features = self.extract_stats_features_torch(ts, axis=-1)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features.cpu()
