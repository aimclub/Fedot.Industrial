"""Utility adapter for multimodal FUTURE encoder stacks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from fedot_ind.core.models.nn.network_impl.encoders.builder import build_encoder
from fedot_ind.core.models.nn.network_impl.encoders.config import EncoderConfig
from fedot_ind.core.models.nn.network_impl.mapping import ENCODER_PRESET_BUILDERS
from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle
from fedot_ind.core.multimodal.enums import MultimodalModality
from fedot_ind.core.multimodal.rules import (
    normalize_modality,
    normalize_unique_modalities,
    validate_modalities_presence,
    validate_registry_supports_modalities,
)


def _count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


class FutureEncoderStack(nn.Module):
    """Multimodal stack that applies one encoder per modality."""

    def __init__(self, encoder_configs: Mapping[MultimodalModality, EncoderConfig]):
        super().__init__()
        if not encoder_configs:
            raise ValueError("FutureEncoderStack requires at least one encoder config.")

        self.encoder_configs = dict(encoder_configs)
        d_models = {config.d_model for config in self.encoder_configs.values()}
        if len(d_models) != 1:
            raise ValueError(
                f"All modality encoders must share the same d_model. Got: {sorted(d_models)}."
            )
        self.embedding_dim = next(iter(d_models))
        self.modalities = tuple(self.encoder_configs.keys())

        self.encoders = nn.ModuleDict(
            {
                modality.value: build_encoder(config)
                for modality, config in self.encoder_configs.items()
            }
        )

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

        embeddings: dict[MultimodalModality, torch.Tensor] = {}
        input_shapes: dict[str, tuple[int, ...]] = {}
        output_shapes: dict[str, tuple[int, ...]] = {}

        for modality in self.modalities:
            tensor = modalities[modality]
            encoder = self.encoders[modality.value]
            embedding = encoder(tensor)
            embeddings[modality] = embedding
            input_shapes[modality.value] = tuple(tensor.shape)
            output_shapes[modality.value] = tuple(embedding.shape)

        if not return_aux:
            return embeddings

        aux = {
            "active_modalities": [modality.value for modality in self.modalities],
            "embedding_dim": self.embedding_dim,
            "num_parameters": {
                "total": _count_parameters(self),
                "per_modality": {
                    modality.value: _count_parameters(self.encoders[modality.value])
                    for modality in self.modalities
                },
            },
            "shapes": {
                "input": input_shapes,
                "output": output_shapes,
            },
        }
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
            registry=ENCODER_PRESET_BUILDERS,
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
            config_map[modality] = ENCODER_PRESET_BUILDERS[modality].build_config(
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
        normalized = {
            normalize_modality(modality): tensor
            for modality, tensor in modalities.items()
        }
        return self.encoder_stack(normalized, return_aux=return_aux)
