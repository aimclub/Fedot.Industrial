import sys
from functools import partial
from itertools import product
from typing import Optional, Union
import pandas as pd
import numpy as np
import torch

from fedot.core.operations.operation_parameters import OperationParameters
from fedot_ind.core.models.base_extractor import BaseExtractor
from fedot_ind.core.operation.transformation.data.point_cloud import TopologicalEmbeddingConfig, PointCloudBuilder, PersistenceConfig, PersistenceDiagramsExtractor
from fedot_ind.core.operation.transformation.representation.topological.topofeatures import TopologicalFeaturesExtractor
from fedot_ind.core.repository.constanst_repository import PERSISTENCE_DIAGRAM_FEATURES

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
        super().__init__(params)
        
        self.point_cloud_builder = None
        self.persistence_extractor = None
        self.feature_extractor = None
        self._current_ts_length = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            multivariate_strategy=self.params.get('multivariate_strategy', 'independent')
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
        
        initialized_features = {
            name: feature_class(max_homology_dim=max_dim)
            for name, feature_class in PERSISTENCE_DIAGRAM_FEATURES.items()
        }
        
        self.feature_extractor = TopologicalFeaturesExtractor(initialized_features)
        self._current_ts_length = ts_length

    def generate_features_from_ts(self, ts_data: Union[pd.DataFrame, np.ndarray, torch.Tensor]) -> pd.DataFrame:
        if isinstance(ts_data, pd.DataFrame):
            ts_data = ts_data.values
        if not isinstance(ts_data, torch.Tensor):
            ts_tensor = torch.tensor(ts_data, dtype=torch.float32, device=self._device)
        else:
            ts_tensor = ts_data.to(self._device)

        if ts_tensor.ndim == 1:
            ts_tensor = ts_tensor.view(1, 1, -1)
        elif ts_tensor.ndim == 2:
            ts_tensor = ts_tensor.unsqueeze(1)

        N = ts_tensor.shape[-1] 

        self._init_pipeline(N)

        point_cloud = self.point_cloud_builder.build(ts_tensor)
        
        diagrams = self.persistence_extractor.transform(point_cloud)
        if isinstance(diagrams, np.ndarray):
            diagrams = torch.tensor(diagrams, dtype=torch.float32, device=self._device)

        features_df = self.feature_extractor.transform(diagrams)
        return features_df
