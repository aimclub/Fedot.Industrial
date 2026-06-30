from typing import Any
import torch

from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle
from fedot_ind.core.multimodal.enums import (
    MultimodalModality,
    NormalizationConfig,
    NormalizationMethod,
)


class MultimodalPreprocessor:
    """Train-aware normalization layer for multimodal time-series bundles."""

    def __init__(
        self,
        normalization_config: NormalizationConfig | None = None,
        *,
        eps: float = 1e-6,
    ) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.normalization_config = (
            normalization_config if normalization_config is not None else {}
        )
        self.eps = eps
        self.fitted_statistics_: dict[str, Any] = {}
        self.is_fitted_ = False

    def fit(self, bundle: MultimodalDataBundle) -> "MultimodalPreprocessor":
        self._validate_bundle(bundle)
        self.fitted_statistics_ = {}

        for modality, steps in self.normalization_config.items():
            current = bundle.modalities[modality]
            self.fitted_statistics_[modality.value] = {
                "steps": [step.value for step in steps],
                "input_shape": tuple(current.shape),
            }
            for step in steps:
                if step is NormalizationMethod.log1p:
                    current = torch.log1p(current.float().clamp_min(0))
                elif step is NormalizationMethod.image_standardization:
                    self._fit_image_standardizer(modality, current)
                    current = self._apply_image_standardizer(modality, current)
            self.fitted_statistics_[modality.value]["output_shape"] = tuple(current.shape)

        self.is_fitted_ = True
        return self

    def transform(self, bundle: MultimodalDataBundle) -> MultimodalDataBundle:
        if not self.is_fitted_:
            raise ValueError("MultimodalPreprocessor must be fitted before transform.")
        self._validate_bundle(bundle)

        modalities = dict(bundle.modalities)
        for modality, steps in self.normalization_config.items():
            current = modalities[modality]
            for step in steps:
                if step is NormalizationMethod.log1p:
                    current = torch.log1p(current.float().clamp_min(0))
                elif step is NormalizationMethod.image_standardization:
                    current = self._apply_image_standardizer(modality, current)
            modalities[modality] = current

        return MultimodalDataBundle(
            modalities=modalities,
            target=bundle.target,
            metadata=dict(bundle.metadata),
        )

    def fit_transform(self, bundle: MultimodalDataBundle) -> MultimodalDataBundle:
        return self.fit(bundle).transform(bundle)

    def _fit_image_standardizer(
        self,
        modality: MultimodalModality,
        tensor: torch.Tensor,
    ) -> None:
        dims = (0, 2, 3) if tensor.ndim == 4 else (0, 1, 2)
        if tensor.ndim not in (3, 4):
            raise ValueError(
                "Image standardization expects a 3D or 4D tensor, "
                f"got shape={tuple(tensor.shape)}."
            )
        mean = tensor.mean(dim=dims, keepdim=True)
        std = tensor.std(dim=dims, unbiased=False, keepdim=True).clamp_min(self.eps)
        self.fitted_statistics_[modality.value]["image_standardization"] = {
            "mean": mean.detach().clone(),
            "std": std.detach().clone(),
            "dims": dims,
            "eps": float(self.eps),
        }

    def _apply_image_standardizer(
        self,
        modality: MultimodalModality,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        stats = self.fitted_statistics_[modality.value]["image_standardization"]
        mean = stats["mean"].to(device=tensor.device, dtype=tensor.dtype)
        std = stats["std"].to(device=tensor.device, dtype=tensor.dtype)
        return torch.nan_to_num(((tensor - mean) / std).float())

    def _validate_bundle(self, bundle: MultimodalDataBundle) -> None:
        if not isinstance(bundle, MultimodalDataBundle):
            raise TypeError(
                f"MultimodalPreprocessor expects MultimodalDataBundle, got {type(bundle)}."
            )
        missing = [
            modality.value
            for modality in self.normalization_config
            if modality not in bundle.modalities
        ]
        if missing:
            raise ValueError(
                f"Bundle does not contain required modalities: {', '.join(missing)}."
            )
