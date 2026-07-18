import sys
from functools import partial
from itertools import product
from typing import Optional, Union
import pandas as pd
import numpy as np
import torch
import warnings
import copy

from fedot.core.operations.operation_parameters import OperationParameters
from fedot_ind.core.models.base_extractor import BaseExtractor
from fedot_ind.core.operation.transformation.data.point_cloud import TopologicalEmbeddingConfig, PointCloudBuilder, PersistenceConfig, PersistenceDiagramsExtractor
from fedot_ind.core.operation.transformation.representation.topological.topofeatures import TopologicalFeaturesExtractor
from fedot_ind.core.repository.constanst_repository import PERSISTENCE_DIAGRAM_FEATURES
from fedot.core.data.data import InputData

sys.setrecursionlimit(1000000000)

class TopologicalExtractor(BaseExtractor):
    """Class for extracting topological features from time series data.

    Args:
        params: parameters for operation

    Example:
        To use this operation you can create pipeline as follows::

            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from fedot_ind.tools.loader import DataLoader
            from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

            train_data, test_data = DataLoader(dataset_name='Ham').load_data()
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('eigen_basis').add_node('topological_extractor').add_node(
                    'rf').build()
                pipeline.fit(input_data)
                features = pipeline.predict(input_data)
                print(features)

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        params = params or {}
        super().__init__(params)
        self._validate_params()

        self.point_cloud_builder = None
        self.persistence_extractor = None
        self.feature_extractor = None
        self._current_ts_length = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.multivariate_strategy = self.params.get('multivariate_strategy', 'independent')

    def _validate_params(self):
        """Validates the hyperparameters of the topological extractor."""
        
        window_share = self.params.get('window_size_as_share', 0.1)
        if not (0 < window_share <= 1):
            warnings.warn(
                f"window_size_as_share should typically be in the range (0, 1]. "
                f"Got: {window_share}. If it is too small, the effective window "
                f"might collapse to 2 points.",
                UserWarning
            )

        stride = self.params.get('stride', 1)
        if not isinstance(stride, int) or stride < 1:
            raise ValueError(f"Stride must be an integer >= 1, got {stride}.")

        delay = self.params.get('delay', 1)
        if not isinstance(delay, int) or delay < 1:
            raise ValueError(f"Delay must be an integer >= 1, got {delay}.")

        valid_strategies = {'independent', 'joint'}
        multivariate_strategy = self.params.get('multivariate_strategy', 'independent')
        if multivariate_strategy not in valid_strategies:
            raise ValueError(
                f"Unsupported multivariate_strategy: '{multivariate_strategy}'. "
                f"Valid options are: {valid_strategies}"
            )

        max_dim = self.params.get('max_homology_dimension', 2)
        if not isinstance(max_dim, int) or max_dim < 0:
            raise ValueError(f"max_homology_dimension must be a non-negative integer, got {max_dim}.")

        valid_filtrations = {'vietoris-rips', 'alpha'}
        filtration = self.params.get('filtration_type', 'vietoris-rips')
        if filtration not in valid_filtrations:
            raise ValueError(
                f"Unsupported filtration_type: '{filtration}'. "
                f"Valid options are: {valid_filtrations}"
            )

        valid_backends = {'gtda', 'ripser++'}
        backend = self.params.get('backend', 'gtda')
        if backend not in valid_backends:
            raise ValueError(
                f"Unsupported backend: '{backend}'. "
                f"Valid options are: {valid_backends}"
            )

        if backend == 'ripser++' and filtration == 'alpha':
            warnings.warn(
                "Methodology mismatch: 'ripser++' backend does not support 'alpha' filtration. "
                "The calculation will be forcefully switched to the 'gtda' backend at runtime.",
                UserWarning
            )

    def _resolve_params(self, ts_length: int) -> tuple[TopologicalEmbeddingConfig, PersistenceConfig]:
        window_share = self.params.get('window_size_as_share', 0.1)
        stride = self.params.get('stride', 1)
        delay = self.params.get('delay', 1)
        filtration = self.params.get('filtration_type', 'vietoris-rips')
        backend = self.params.get('backend', 'gtda')
        
        max_dim = self.params.get('max_homology_dimension', 2)
        homology_dims = tuple(range(max_dim + 1))
        
        absolute_window = max(2, int(ts_length * window_share))

        embed_config = TopologicalEmbeddingConfig(
            window_size=absolute_window,
            stride=stride,
            delay=delay,
            multivariate_strategy=self.multivariate_strategy
        )

        pers_config = PersistenceConfig(
            homology_dimensions=homology_dims,
            backend=backend,
            filtration_type=filtration,
            normalize=self.params.get('normalize', True),
            distance_device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        return embed_config, pers_config

    def _init_pipeline(self, ts_length: int):
        """Initialize the topological feature extraction pipeline if the time series length has changed."""
        if self._current_ts_length == ts_length and self.point_cloud_builder is not None:
            return  

        embed_config, pers_config = self._resolve_params(ts_length)

        self.point_cloud_builder = PointCloudBuilder(embed_config)
        self.persistence_extractor = PersistenceDiagramsExtractor(pers_config)

        max_dim = max(pers_config.homology_dimensions)
        
        initialized_features = {}
        for name, feature_class in PERSISTENCE_DIAGRAM_FEATURES.items():
            feat_instance = feature_class.__class__(max_homology_dim=max_dim)
            initialized_features[name] = feat_instance
        
        self.feature_extractor = TopologicalFeaturesExtractor(initialized_features)
        self._current_ts_length = ts_length
    
    @torch.no_grad()
    def generate_features_from_ts(self, ts_data: Union[pd.DataFrame, np.ndarray, torch.Tensor]) -> pd.DataFrame:
        """
        Extracts topological features from time series data.
        
        Args:
            ts_data: Input time series data. 
                The expected shape is (B, C, N), where:
                B - batch size (number of time series),
                C - number of channels (variables),
                N - length of the time series.
                
        Returns:
            pd.DataFrame: Extracted topological features.
        """

        if hasattr(ts_data, 'features'):
            ts_data = ts_data.features
        if isinstance(ts_data, pd.DataFrame):
            ts_data = ts_data.values
        if not isinstance(ts_data, torch.Tensor):
            ts_tensor = torch.tensor(ts_data, dtype=torch.float32, device=self._device)
        else:
            ts_tensor = ts_data.to(self._device)

        if ts_tensor.ndim != 3:
            warnings.warn(
                f"Expected input tensor of shape (B, C, N), but got {ts_tensor.ndim}D tensor "
                f"with shape {tuple(ts_tensor.shape)}. The tensor will be automatically reshaped, "
                f"which may lead to unexpected behavior if the dimensions are misaligned.",
                UserWarning,
                stacklevel=2
            )
            if ts_tensor.ndim == 1:
                ts_tensor = ts_tensor.view(1, 1, -1)
            elif ts_tensor.ndim == 2:
                ts_tensor = ts_tensor.unsqueeze(1)
            else:
                raise ValueError(f"Expected <= 3 dimensions (B, C, N), got {ts_tensor.ndim}")

        N = ts_tensor.shape[-1] 

        self._init_pipeline(N)

        point_cloud = self.point_cloud_builder.build(ts_tensor)
        if self.multivariate_strategy == 'independent':
            B, C, M, W = point_cloud.shape
            point_cloud = point_cloud.reshape(B * C, M, W)
        diagrams = self.persistence_extractor.transform(point_cloud)

        if isinstance(diagrams, np.ndarray):
            diagrams = torch.tensor(diagrams, dtype=torch.float32, device=self._device)

        features_tensor, column_names = self.feature_extractor.transform(diagrams)
        
        if self.multivariate_strategy == 'independent':
            features_tensor = features_tensor.view(B, -1)
            column_names = [f"{name}_ch{ch}" for ch in range(C) for name in column_names]
        return pd.DataFrame(features_tensor.cpu().numpy(), columns=column_names)
    

    def fit(self, input_data: InputData):
        self.is_fitted = True
        return self
    
    def _transform(self, input_data: InputData) -> np.ndarray:
        if input_data.features is None or len(input_data.features) == 0:
            raise ValueError("Input data features are empty.")
        
        features_df = self.generate_features_from_ts(input_data.features)

        feature_matrix = features_df.values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        self.predict = feature_matrix
        
        return self.predict