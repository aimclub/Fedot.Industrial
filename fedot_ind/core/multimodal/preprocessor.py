from typing import Any

from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle
from fedot_ind.core.multimodal.enums import (
    MultimodalModality,
    NormalizationConfig,
)
from fedot_ind.core.multimodal.mapping import NORMALIZATION_HANDLERS


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
        self._handlers_: dict[str, dict[str, Any]] = {}
        self.is_fitted_ = False

    def fit(self, bundle: MultimodalDataBundle) -> "MultimodalPreprocessor":
        self._validate_bundle(bundle)
        self._handlers_ = {}

        for modality, steps in self.normalization_config.items():
            current = bundle.modalities[modality]
            modality_handlers: list[tuple[str, Any]] = []
            for step in steps:
                handler_cls = NORMALIZATION_HANDLERS.get(step)
                if handler_cls is None:
                    raise ValueError(f"Unsupported normalization step: {step}.")
                handler = handler_cls(eps=self.eps)
                handler.fit(current)
                current = handler.transform(current)
                modality_handlers.append((step.value, handler))
            self._handlers_[modality.value] = {
                "steps": modality_handlers,
                "input_shape": tuple(bundle.modalities[modality].shape),
                "output_shape": tuple(current.shape),
            }

        self.is_fitted_ = True
        return self

    def transform(self, bundle: MultimodalDataBundle) -> MultimodalDataBundle:
        if not self.is_fitted_:
            raise ValueError("MultimodalPreprocessor must be fitted before transform.")
        self._validate_bundle(bundle)

        modalities = dict(bundle.modalities)
        for modality, steps in self.normalization_config.items():
            current = modalities[modality]
            modality_handlers = self._handlers_.get(modality.value)
            if modality_handlers is None:
                raise ValueError(
                    f"Missing fitted handlers for modality '{modality.value}'."
                )
            handlers = modality_handlers["steps"]
            if len(handlers) != len(tuple(steps)):
                raise ValueError(
                    f"Handler count mismatch for modality '{modality.value}'."
                )
            for _, handler in handlers:
                current = handler.transform(current)
            modalities[modality] = current

        return MultimodalDataBundle(
            modalities=modalities,
            target=bundle.target,
            metadata=dict(bundle.metadata),
        )

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
