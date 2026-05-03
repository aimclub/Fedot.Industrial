from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional, Union

import numpy as np
import torch
from fedot.core.data.data import InputData

from fedot_ind.core.operation.transformation.torch_backend.image_transformation.transformation_mapping import (
    TRANSFORMATION_MAPPING,
)
from fedot_ind.core.operation.transformation.torch_backend.image_transformation.types import (
    ImageTransformationType,
)


class ImageTransformer:
    """
    Example:
        params = {
            'transformation_type': ImageTransformationType.MTF,
            'transfromation_params': {
                'image_size': 1.0,
                'n_bins': 8,
                'strategy': 'quantile',
            }
        }
    """

    def __init__(self, params: Optional[dict[str, Any]] = None):
        params = params or {}
        self.tranfromation_type = params.get(
            "transformation_type", ImageTransformationType.MTF
        )
        self.transfromation_params = params.get("transfromation_params", {})

    def _prepare_features(
        self, features: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(features, np.ndarray):
            features = torch.as_tensor(features, dtype=torch.float64)
        if not isinstance(features, torch.Tensor):
            raise TypeError(
                f"features must be ndarray or Tensor, got {type(features)!r}"
            )
        if features.ndim == 1:
            return features.unsqueeze(0)
        if features.ndim == 2:
            return features
        if features.ndim == 3:
            b, c, t = features.shape
            return features.reshape(b * c, t)
        raise ValueError(
            f"Expected features with ndim 1–3, got shape {tuple(features.shape)}"
        )

    def _transform(self, input_data: InputData) -> torch.Tensor:
        """
        Args:
            input_data: Fedot ``InputData`` или любой объект с полем ``features``.

        Returns:
            Tensor ``(batch, …)`` в зависимости от выбранного преобразования.
        """
        features = self._prepare_features(input_data.features)
        cls = TRANSFORMATION_MAPPING.get(self.tranfromation_type)
        if cls is None:
            raise ValueError(f"Unknown transformation: {self.tranfromation_type!r}")
        transformer = cls(params=self.transfromation_params)
        return transformer.transform(features)

    def transform(
        self, features: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """Публичный вход: массив или тензор без обёртки ``InputData``."""
        return self._transform(SimpleNamespace(features=features))
