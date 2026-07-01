from typing import Optional, Any
import inspect
import torch

from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot_ind.core.models.base_extractor import BaseExtractor
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.operation.transformation.torch_backend.enums import (
    STAT_FEATURE_CONFIG,
    StatisticalFeature,
)
from fedot_ind.core.repository.constanst_repository import (
    STAT_METHODS_GLOBAL_TORCH,
    STAT_METHODS_TORCH,
)
from fedot_ind.core.operation.transformation.torch_backend.statistical.tools import (
    build_default_feature_config,
    method_registry,
    normalize_feature_key,
)


DEFAULT_CONFIG = build_default_feature_config(STAT_METHODS_TORCH)
DEFAULT_GLOBAL_CONFIG = build_default_feature_config(
    STAT_METHODS_GLOBAL_TORCH
)


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
        params = params or OperationParameters()
        super().__init__(params)
        self.window_size = params.get('window_size', 0)
        self.stride = params.get('stride', 1)
        self.add_global_features = params.get('add_global_features', True)
        self.logging_params.update({'Wsize': self.window_size,
                                    'Stride': self.stride})
        self.config = self._normalize_feature_config(
            params.get('stat_feature_config', DEFAULT_CONFIG)
        )
        self.global_config = self._normalize_feature_config(
            params.get('stat_feature_global_config', DEFAULT_GLOBAL_CONFIG)
        )
        self._methods_local = method_registry(STAT_METHODS_TORCH)
        self._methods_global = method_registry(STAT_METHODS_GLOBAL_TORCH)

    @staticmethod
    def _normalize_feature_config(config: Any) -> STAT_FEATURE_CONFIG:
        normalized: STAT_FEATURE_CONFIG = {}
        if not config:
            return normalized
        for raw_key, raw_params in dict(config).items():
            feature = normalize_feature_key(raw_key)
            normalized[feature] = dict(raw_params or {})
        return normalized

    @staticmethod
    def _supports_kwargs(method: callable) -> bool:
        signature = inspect.signature(method)
        return any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    @staticmethod
    def _apply_feature_method(
        method: callable,
        time_series: torch.Tensor,
        axis: int,
        kwargs: dict[str, Any],
        supports_kwargs: bool,
    ):
        if supports_kwargs:
            return method(time_series, axis=axis, **kwargs)
        return method(time_series, axis)

    def _select_methods(self, add_global_features: bool) -> list[tuple[callable, dict[str, Any], bool]]:
        method_map = self._methods_global if add_global_features else self._methods_local
        config = self.global_config if add_global_features else self.config
        selected = []
        for feature, feature_kwargs in config.items():
            method = method_map.get(feature)
            if method is not None:
                selected.append((method, dict(feature_kwargs), self._supports_kwargs(method)))
        return selected

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

    def generate_features_from_ts(self, ts: torch.Tensor, ) -> torch.Tensor:
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

    def apply_window_for_stat_feature_torch(self,
                                            ts_data: torch.Tensor,
                                            feature_generator: callable,
                                            window_size: int = None) -> torch.Tensor:
        """
        Method for creating windows and extracting base statistical features
        for a given time series or batch of time series.
        """
        axis = ts_data.ndim - 1
        if window_size is None:
            window_size = round(ts_data.shape[axis] / 10)
        else:
            window_size = round(ts_data.shape[axis] * (window_size / 100))
        window_size = max(window_size, 5)

        if self.use_sliding_window:
            if self.stride > 1:
                subseq_set = HankelMatrix(time_series=ts_data,
                                          window_size=window_size,
                                          strides=self.stride).trajectory_matrix
            else:
                window_length = ts_data.shape[axis] - window_size
                subseq_set = ts_data.unfold(
                    dimension=axis,
                    size=window_length,
                    step=self.stride
                )
            if subseq_set.ndim > 2:
                subseq_set = subseq_set.transpose(1, 2)
            else:
                subseq_set = subseq_set.T
        else:
            T = ts_data.shape[1]
            num_windows = T // window_size
            T_eff = num_windows * window_size
            ts_cut = ts_data[:, :T_eff]
            subseq_set = ts_cut.reshape(2, num_windows, window_size)
        features = feature_generator(subseq_set)
        features = torch.stack(features, dim=0).to(ts_data.device)
        if features.ndim > 2:
            features = features.permute(1, 2, 0)
        else:
            features = features.T
        return features

    def get_statistical_features_torch(
            self,
            time_series: torch.Tensor,
            add_global_features: bool = False,
            axis: int = -1,
            max_elements: int = 50000000) -> tuple:
        """
        Method for creating baseline statistical features for a given time series.
        Creates batches, if number of values in time series is more than max_elements.

        Args:
            add_global_features: if True, global features are added to the feature set
            time_series: time series for which features are generated
            max_elements: threshold for using batches

        Returns:
            tuple: features

        """
        if self.is_multichanel:
            if time_series.ndim == 3:
                batch = time_series.shape[0]
                time_series = time_series.reshape(batch, -1)
            elif time_series.ndim > 3:
                batch, b, *rest = time_series.shape
                time_series = time_series.reshape(batch, b, -1)

        if add_global_features:
            list_of_methods = self._select_methods(add_global_features=True)
        else:
            list_of_methods = self._select_methods(add_global_features=False)

        if time_series.numel() <= max_elements:
            return [
                self._apply_feature_method(
                    method,
                    time_series,
                    axis,
                    kwargs,
                    supports_kwargs,
                )
                for method, kwargs, supports_kwargs in list_of_methods
            ]
        B = time_series.shape[0]
        elems_per_sample = time_series[0].numel()
        batch_size = max(1, max_elements // elems_per_sample)
        accumulators = [[] for _ in list_of_methods]
        for start in range(0, B, batch_size):
            end = min(start + batch_size, B)
            ts_batch = time_series[start:end]
            batch_results = [
                self._apply_feature_method(
                    method,
                    ts_batch,
                    axis,
                    kwargs,
                    supports_kwargs,
                )
                for method, kwargs, supports_kwargs in list_of_methods
            ]
            for i, res in enumerate(batch_results):
                accumulators[i].append(res)

        merged = []
        for parts in accumulators:
            if isinstance(parts[0], (float, int)):
                merged.append(
                    torch.tensor(parts) if isinstance(parts[0], torch.Tensor) else parts
                )
            else:
                merged.append(torch.cat(parts, dim=0))
        return merged
