from __future__ import annotations

from typing import Any
from typing import Protocol

import torch

from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle
from fedot_ind.core.multimodal.enums import NormalizationConfig
from fedot_ind.core.multimodal.mapping import NORMALIZATION_HANDLERS
from fedot_ind.core.multimodal.rules import (
    validate_bundle_type,
    validate_modalities_presence,
    validate_positive_number,
    validate_registry_supports_normalization_steps,
)


class NormalizerProtocol(Protocol):
    """Runtime contract for train-aware normalization handlers."""

    def fit(self, X: torch.Tensor) -> None:
        """Fit handler statistics on train data."""

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Transform data using fitted statistics."""

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted statistics."""


class MultimodalPreprocessor:
    """Train-aware normalization layer for multimodal time-series bundles."""

    def __init__(
        self,
        normalization_config: NormalizationConfig | None = None,
        *,
        eps: float = 1e-6,
    ) -> None:
        validate_positive_number("eps", eps)
        self.normalization_config = (
            normalization_config if normalization_config is not None else {}
        )
        validate_registry_supports_normalization_steps(
            self.normalization_config,
            NORMALIZATION_HANDLERS,
        )
        self.eps = eps
        self._handlers_: dict[str, dict[str, Any]] = {}
        self.is_fitted_ = False

    def fit(self, bundle: MultimodalDataBundle) -> "MultimodalPreprocessor":
        self._validate_bundle(bundle)
        self._fit_pipeline(bundle)
        self.is_fitted_ = True
        return self

    def _fit_pipeline(self, bundle: MultimodalDataBundle) -> None:
        """Fit configured handlers and pass train data between pipeline steps."""

        self._handlers_ = {}

        for modality, steps in self.normalization_config.items():
            current = bundle.modalities[modality]
            modality_handlers: list[tuple[str, NormalizerProtocol]] = []
            for step in steps:
                handler = NORMALIZATION_HANDLERS[step](eps=self.eps)
                handler.fit(current)
                current = handler.transform(current)
                modality_handlers.append((step.value, handler))
            self._handlers_[modality.value] = {
                "steps": modality_handlers,
                "input_shape": tuple(bundle.modalities[modality].shape),
                "output_shape": tuple(current.shape),
            }

    def transform(self, bundle: MultimodalDataBundle) -> MultimodalDataBundle:
        if not self.is_fitted_:
            raise ValueError("MultimodalPreprocessor must be fitted before transform.")
        self._validate_bundle(bundle)

        modalities = dict(bundle.modalities)
        for modality in self.normalization_config:
            current = modalities[modality]
            for _, handler in self._handlers_[modality.value]["steps"]:
                current = handler.transform(current)
            modalities[modality] = current

        return bundle.with_modalities(modalities)

    def fit_transform(self, bundle: MultimodalDataBundle) -> MultimodalDataBundle:
        return self.fit(bundle).transform(bundle)

    @property
    def fitted_statistics_(self) -> dict[str, Any]:
        statistics: dict[str, Any] = {}
        for modality, payload in self._handlers_.items():
            step_entries = payload.get("steps", [])
            statistics[modality] = {
                "steps": [step_name for step_name, _ in step_entries],
                "input_shape": payload.get("input_shape"),
                "output_shape": payload.get("output_shape"),
            }
            for step_name, handler in step_entries:
                statistics[modality][step_name] = handler.get_state()
        return statistics

    def _validate_bundle(self, bundle: MultimodalDataBundle) -> None:
        validate_bundle_type(bundle, MultimodalDataBundle)
        validate_modalities_presence(
            required=self.normalization_config,
            available=bundle.modalities,
            source_label="Bundle",
        )
