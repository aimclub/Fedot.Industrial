from typing import Optional

import dask
import numpy as np
import torch

from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot_ind.core.models.base_extractor import BaseExtractor
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.operation.transformation.data.kernel_matrix import TorchTSTransformer
from fedot_ind.core.operation.transformation.torch_backend.recurrence.sequences import RecurrenceFeatureExtractorTorch


class RecurrenceExtractor(BaseExtractor):
    """Class responsible for wavelet feature generator experiment.

    Attributes:
        transformer: TorchTSTransformer object.
        self.extractor: RecurrenceFeatureExtractorTorch object.
        self.rec_metric: str, the metric for calculating the recurrence matrix.
        self.window_size: int, the window size.
        self.image_mode: bool, if True, then created 3D recurrence matrix.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.window_size = params.get('window_size', 0)
        self.stride = params.get('stride', 1)
        # TODO add threshold for other metrics
        self.rec_metric = params.get('rec_metric', 'cosine')
        self.image_mode = params.get('image_mode', False)
        self.transformer = TorchTSTransformer
        self.extractor = RecurrenceFeatureExtractorTorch

    def __repr__(self):
        return 'Reccurence Class for TS representation'

    def _generate_features_from_ts(self, ts: torch.Tensor) -> torch.Tensor:
        if self.window_size != 0:
            trajectory_transformer = HankelMatrix(time_series=ts,
                                                  window_size=self.window_size,
                                                  strides=self.stride)
            ts = trajectory_transformer.trajectory_matrix
            self.ts_length = trajectory_transformer.ts_length
        specter = self.transformer(time_series=ts,
                                   rec_metric=self.rec_metric)
        
        if not self.image_mode:
            feature_df = specter.ts_to_recurrence_matrix()
            feature_df = self.extractor(
                recurrence_matrix=feature_df).quantification_analysis()
            features = torch.tensor(list(feature_df.values()))
        else:
            features = specter.ts_to_3d_recurrence_matrix()
        return features

    def generate_recurrence_features(self, ts: torch.Tensor):
        if ts.ndim < 3:
            aggregation_df = self._generate_features_from_ts(ts)
        else:
            aggregation_df = self._get_torch_feature_matrix(
                self._generate_features_from_ts, ts)
        return aggregation_df
