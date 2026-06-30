from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import prod
from typing import Any, Sequence

import torch

from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle
from fedot_ind.core.multimodal.enums import MultimodalModality


@dataclass
class MultimodalPreprocessor:
    """Train-aware preparation layer for multimodal time-series models.

    The initial implementation exposes only the raw modality. The public
    contract is intentionally shaped for adding train-fitted stats/image
    modalities without changing benchmark data loading.

    ``X`` and ``y`` must already be ``torch.Tensor`` instances. This class
    only reshapes inputs to ``(batch, channels, timestamps)`` and applies
    modality-specific normalization.
    """

    modalities: Sequence[MultimodalModality | str] = (MultimodalModality.raw,)
    torch_device: Any = "auto"
    dtype: torch.dtype = torch.float32
    eps: float = 1e-8
    transform_params: dict[MultimodalModality | str, dict[str, Any]] = field(
        default_factory=dict
    )
    fitted_statistics_: dict[str, Any] = field(default_factory=dict, init=False)
    is_fitted_: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.modalities = tuple(
            self._normalize_modality(modality) for modality in self.modalities
        )
        self.transform_params = {
            self._normalize_modality(name): dict(params)
            for name, params in self.transform_params.items()
        }
        if self.eps <= 0:
            raise ValueError("eps must be positive.")
        if not self.modalities:
            raise ValueError("MultimodalPreprocessor requires at least one modality.")

    def fit(self, X: torch.Tensor, y: torch.Tensor | None = None) -> "MultimodalPreprocessor":
        del y
        raw = self._reshape_raw(X)
        normalized_raw = self._normalize_raw(raw)
        self.fitted_statistics_ = {
            "raw": {
                "normalization": "per_sample_z_norm",
                "eps": float(self.eps),
                "train_input_shape": tuple(int(size) for size in raw.shape),
                "train_output_shape": tuple(int(size) for size in normalized_raw.shape),
            }
        }
        self.is_fitted_ = True
        return self

    def transform(
        self,
        X: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> MultimodalDataBundle:
        if not self.is_fitted_:
            raise ValueError("MultimodalPreprocessor must be fitted before transform.")

        raw = self._reshape_raw(X)
        modalities: dict[MultimodalModality, torch.Tensor] = {}
        if MultimodalModality.raw in self.modalities:
            modalities[MultimodalModality.raw] = self._normalize_raw(raw)

        metadata = {
            "fitted_statistics": deepcopy(self.fitted_statistics_),
            "transform_params": self._metadata_transform_params(),
            "preprocessor": {
                "name": self.__class__.__name__,
                "modalities": [modality.value for modality in self.modalities],
                "eps": float(self.eps),
            },
        }
        return MultimodalDataBundle(
            modalities=modalities,
            target=self._validate_target(y, n_samples=raw.shape[0]),
            metadata=metadata,
        )

    def fit_transform(
        self,
        X: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> MultimodalDataBundle:
        self.fit(X, y)
        return self.transform(X, y)

    def _reshape_raw(self, values: torch.Tensor) -> torch.Tensor:
        if not isinstance(values, torch.Tensor):
            raise TypeError(
                f"X must be torch.Tensor, got {type(values)}."
            )

        if values.ndim == 0:
            raise ValueError("X must contain at least one sample.")
        if values.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        if values.ndim == 1:
            return values.reshape(1, 1, -1)
        if values.ndim == 2:
            return values.reshape(values.shape[0], 1, values.shape[1])
        if values.ndim == 3:
            return values
        return values.reshape(
            values.shape[0],
            prod(values.shape[1:-1]),
            values.shape[-1],
        )

    def _normalize_raw(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=-1, keepdim=True)
        std = values.std(dim=-1, keepdim=True, unbiased=False)
        scale = torch.where(std > self.eps, std, torch.ones_like(std))
        return (values - mean) / scale

    def _validate_target(
        self,
        values: torch.Tensor | None,
        *,
        n_samples: int,
    ) -> torch.Tensor | None:
        if values is None:
            return None
        if not isinstance(values, torch.Tensor):
            raise TypeError(
                f"Target must be torch.Tensor or None, got {type(values)}."
            )
        if values.ndim == 0:
            raise ValueError("Target must have sample dimension.")
        if int(values.shape[0]) != int(n_samples):
            raise ValueError(
                "Target and modalities must have the same number of samples. "
                f"Got target size {int(values.shape[0])}, modalities size {n_samples}."
            )
        return values

    def _metadata_transform_params(self) -> dict[MultimodalModality, dict[str, Any]]:
        params = {name: dict(values) for name, values in self.transform_params.items()}
        params.setdefault(
            MultimodalModality.raw,
            {
                "normalization": "per_sample_z_norm",
                "eps": float(self.eps),
            },
        )
        return params

    @staticmethod
    def _normalize_modality(value: MultimodalModality | str) -> MultimodalModality:
        if isinstance(value, MultimodalModality):
            return value
        try:
            return MultimodalModality(str(value))
        except ValueError as exc:
            raise ValueError(f"Unsupported multimodal modality: {value!r}") from exc
