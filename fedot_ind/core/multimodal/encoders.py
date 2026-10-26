"""Utility adapter for multimodal FUTURE encoder stacks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from fedot_ind.neural.encoders.config import EncoderConfig
from fedot_ind.neural.encoders.stack import EncoderStack
from fedot_ind.core.multimodal.encoder_rules import build_encoder_config_map
from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle
from fedot_ind.core.multimodal.enums import MultimodalModality
from fedot_ind.core.multimodal.mapping import MODALITY_CAPABILITIES
from fedot_ind.core.multimodal.rules import (
    normalize_modality,
    normalize_unique_modalities,
    validate_modalities_presence,
    validate_registry_supports_modalities,
)


class FutureEncoderStack(EncoderStack):
    """Domain wiring over the neutral stack, without a state_dict prefix."""

    def __init__(self, encoder_configs: Mapping[MultimodalModality, EncoderConfig]):
        if not encoder_configs:
            raise ValueError("FutureEncoderStack requires at least one encoder config.")
        configs = build_encoder_config_map(list(encoder_configs.items()))
        super().__init__({modality.value: config for modality, config in configs.items()})
        self.encoder_configs = configs
        self.modalities = tuple(configs)

    def __setstate__(self, state):
        super().__setstate__(state)
        # Legacy full-module checkpoints predate the neutral stream metadata.
        # Restore descriptions only; never rebuild layers or draw new weights.
        if not hasattr(self, "stream_names"):
            self.stream_names = tuple(modality.value for modality in self.modalities)
        if not hasattr(self, "_stream_configs"):
            self._stream_configs = {modality.value: config for modality, config in self.encoder_configs.items()}

    def forward(
        self,
        modalities: Mapping[MultimodalModality, torch.Tensor],
        return_aux: bool = False,
    ) -> dict[MultimodalModality, torch.Tensor] | tuple[
        dict[MultimodalModality, torch.Tensor], dict[str, Any]
    ]:
        """Encode already-normalized modalities.

        Keys must be MultimodalModality; MultimodalDataBundle guarantees that,
        and FutureMultimodalEncoderAdapter.encode_modalities coerces the loose
        mappings that come from callers.
        """

        validate_modalities_presence(
            required=self.modalities,
            available=modalities,
            source_label="Encoder stack input",
        )

        result = super().forward(
            {modality.value: modalities[modality] for modality in self.modalities},
            return_aux=return_aux,
        )
        if not return_aux:
            return {modality: result[modality.value] for modality in self.modalities}
        streams, aux = result
        embeddings = {modality: streams[modality.value] for modality in self.modalities}
        aux["active_modalities"] = aux.pop("active_streams")
        aux["num_parameters"]["per_modality"] = aux["num_parameters"].pop("per_stream")
        return embeddings, aux


class FutureMultimodalEncoderAdapter(nn.Module):
    """Utility that builds family-based encoders from multimodal bundles.

    This class is intentionally not a FEDOT operation adapter: it exposes
    deterministic encoder-stack utilities for FUTURE model code.
    """

    def __init__(self, params: Mapping[str, Any] | None = None):
        super().__init__()
        self.params = dict(params or {})
        self.d_model = int(self.params.get("d_model", 128))
        self.encoder_stack: FutureEncoderStack | None = None

    def configure_from_bundle(
        self,
        bundle: MultimodalDataBundle,
        modalities: Sequence[MultimodalModality | str] | None = None,
        encoder_kwargs: Mapping[str, Any] | None = None,
    ) -> FutureEncoderStack:
        if modalities is None:
            selected_modalities = tuple(bundle.available_modalities)
        else:
            selected_modalities = normalize_unique_modalities(modalities)

        validate_registry_supports_modalities(
            modalities=selected_modalities,
            registry=MODALITY_CAPABILITIES,
            registry_label="encoder adapter",
        )
        validate_modalities_presence(
            required=selected_modalities,
            available=bundle.modalities,
            source_label="Bundle",
        )

        kwargs_map = dict(encoder_kwargs or self.params.get("encoder_kwargs", {}))
        config_map: dict[MultimodalModality, EncoderConfig] = {}
        for modality in selected_modalities:
            shape = bundle.shapes[modality]
            modality_kwargs = dict(kwargs_map.get(modality.value, {}))
            config_map[modality] = MODALITY_CAPABILITIES[
                modality
            ].encoder_preset.build_config(
                shape=shape,
                d_model=self.d_model,
                kwargs=modality_kwargs,
            )

        self.encoder_stack = FutureEncoderStack(config_map)
        return self.encoder_stack

    def encode_bundle(
        self,
        bundle: MultimodalDataBundle,
        return_aux: bool = False,
    ) -> dict[MultimodalModality, torch.Tensor] | tuple[
        dict[MultimodalModality, torch.Tensor], dict[str, Any]
    ]:
        if self.encoder_stack is None:
            self.configure_from_bundle(bundle=bundle)
        assert self.encoder_stack is not None
        return self.encoder_stack(bundle.modalities, return_aux=return_aux)

    def encode_modalities(
        self,
        modalities: Mapping[MultimodalModality | str, torch.Tensor],
        return_aux: bool = False,
    ) -> dict[MultimodalModality, torch.Tensor] | tuple[
        dict[MultimodalModality, torch.Tensor], dict[str, Any]
    ]:
        if self.encoder_stack is None:
            raise ValueError(
                "Encoder stack is not configured. Call configure_from_bundle first."
            )
        normalized = {}
        for raw_modality, tensor in modalities.items():
            modality = normalize_modality(raw_modality)
            if modality in normalized:
                raise ValueError(f"Duplicate modality definition: {modality.value}.")
            normalized[modality] = tensor
        return self.encoder_stack(normalized, return_aux=return_aux)
