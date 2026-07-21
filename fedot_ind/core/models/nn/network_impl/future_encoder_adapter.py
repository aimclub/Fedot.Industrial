"""Thin BaseNeuralModel adapter for multimodal FUTURE encoders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.models.nn.network_impl.base_nn_model import BaseNeuralModel
from fedot_ind.core.models.nn.network_impl.encoders.builder import build_encoder
from fedot_ind.core.models.nn.network_impl.encoders.config import EncoderConfig
from fedot_ind.core.models.nn.models_rules import normalize_modality
from fedot_ind.core.models.nn.network_impl.mapping import ENCODER_PRESET_BUILDERS
from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle
from fedot_ind.core.multimodal.enums import MultimodalModality


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
        modalities: Mapping[MultimodalModality | str, torch.Tensor],
        return_aux: bool = False,
    ) -> dict[MultimodalModality, torch.Tensor] | tuple[
        dict[MultimodalModality, torch.Tensor], dict[str, Any]
    ]:
        normalized = {
            normalize_modality(modality): tensor
            for modality, tensor in modalities.items()
        }
        missing = [modality for modality in self.modalities if modality not in normalized]
        if missing:
            missing_values = [modality.value for modality in missing]
            raise ValueError(
                f"Missing required modalities for encoder stack: {missing_values}."
            )

        embeddings: dict[MultimodalModality, torch.Tensor] = {}
        input_shapes: dict[str, tuple[int, ...]] = {}
        output_shapes: dict[str, tuple[int, ...]] = {}

        for modality in self.modalities:
            tensor = normalized[modality]
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


class FutureMultimodalEncoderAdapter(BaseNeuralModel):
    """Thin adapter that builds family-based encoders from multimodal bundles."""

    def __init__(self, params: OperationParameters | None = None):
        super().__init__(params)
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
            selected_modalities = self._normalize_unique_modalities(modalities)

        unsupported = [
            modality.value
            for modality in selected_modalities
            if modality not in ENCODER_PRESET_BUILDERS
        ]
        if unsupported:
            raise ValueError(
                f"Unsupported modalities for encoder adapter: {sorted(unsupported)}."
            )

        kwargs_map = dict(encoder_kwargs or self.params.get("encoder_kwargs", {}))
        config_map: dict[MultimodalModality, EncoderConfig] = {}
        for modality in selected_modalities:
            if modality not in bundle.modalities:
                raise ValueError(
                    f"Bundle does not contain requested modality '{modality.value}'."
                )
            shape = bundle.shapes[modality]
            modality_kwargs = dict(kwargs_map.get(modality.value, {}))
            config_map[modality] = self._build_preset_config(
                modality=modality,
                shape=shape,
                modality_kwargs=modality_kwargs,
            )

        self.encoder_stack = FutureEncoderStack(config_map)
        self.model = self.encoder_stack
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
        return self.encoder_stack(modalities, return_aux=return_aux)

    def _build_preset_config(
        self,
        modality: MultimodalModality,
        shape: tuple[int, ...],
        modality_kwargs: dict[str, Any],
    ) -> EncoderConfig:
        preset_entry = ENCODER_PRESET_BUILDERS.get(modality)
        if preset_entry is None:
            raise ValueError(f"Unknown modality '{modality.value}'.")

        preset_builder, shape_param = preset_entry
        return preset_builder(
            d_model=self.d_model,
            **{shape_param: int(shape[1])},
            **modality_kwargs,
        )

    @staticmethod
    def _normalize_unique_modalities(
        modalities: Sequence[MultimodalModality | str],
    ) -> tuple[MultimodalModality, ...]:
        normalized: list[MultimodalModality] = []
        seen: set[MultimodalModality] = set()
        for modality in modalities:
            normalized_modality = normalize_modality(modality)
            if normalized_modality in seen:
                raise ValueError(
                    f"Duplicate modality in adapter configuration: {normalized_modality.value}."
                )
            seen.add(normalized_modality)
            normalized.append(normalized_modality)
        if not normalized:
            raise ValueError("Adapter modality list must not be empty.")
        return tuple(normalized)

    def _init_model(self, ts) -> tuple:
        raise NotImplementedError(
            "FutureMultimodalEncoderAdapter exposes encoder stack utilities only."
        )
