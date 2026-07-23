"""Composable multimodal classifier for FUTURE fusion methods."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from fedot_ind.core.models.future.enums import FusionMethod
from fedot_ind.core.models.future.mapping import (
    FUSION_REGISTRY,
    FusionRegistryEntry,
)
from fedot_ind.core.models.future.rules import (
    normalize_unique_modalities,
    require_initialized_model_parts,
    require_resolved_modalities,
    validate_context_modalities_for_raw_centered,
    validate_encoder_registry_has_modalities,
    validate_modalities_presence,
    validate_multimodal_bundle_input,
    validate_positive_int,
    validate_supported_fusion_method,
)
from fedot_ind.core.models.nn.network_impl.encoders.builder import build_encoder
from fedot_ind.core.models.nn.network_impl.mapping import ENCODER_PRESET_BUILDERS
from fedot_ind.core.models.nn.network_modules.activation import get_activation_fn
from fedot_ind.core.models.nn.network_modules.layers.linear_layers import LinLnDrop
from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle
from fedot_ind.core.multimodal.enums import MultimodalModality
from fedot_ind.core.models.future.tools import (
    AuxOutputConfig,
    FusionAuxOutput,
    count_parameters,
)


class ConfigurableMultimodalFusionClassifier(nn.Module):
    """Multimodal classifier composed from encoders, fusion strategy and head."""

    def __init__(
        self,
        num_classes: int,
        fusion_method: FusionMethod | str,
        d_model: int = 128,
        modalities: Sequence[MultimodalModality | str] | None = None,
        encoder_kwargs: Mapping[str, dict[str, Any]] | None = None,
        fusion_kwargs: Mapping[str, Any] | None = None,
        head_hidden_dim: int | None = None,
        head_dropout: float = 0.2,
        head_activation: str = "GELU",
        raw_modality: MultimodalModality | str = MultimodalModality.raw,
        aux_output_config: AuxOutputConfig | None = None,
        auto_build_on_forward: bool = False,
    ):
        super().__init__()

        validate_positive_int(name="num_classes", value=num_classes, min_value=1)
        validate_positive_int(name="d_model", value=d_model, min_value=1)
        if head_hidden_dim is not None:
            validate_positive_int(
                name="head_hidden_dim",
                value=head_hidden_dim,
                min_value=1,
            )

        self.fusion_method = fusion_method
        self.fusion_method = validate_supported_fusion_method(
            fusion_method=fusion_method,
            fusion_registry=FUSION_REGISTRY,
        )
        self.fusion_entry: FusionRegistryEntry = FUSION_REGISTRY[self.fusion_method]

        self.modalities: tuple[MultimodalModality, ...] | None = None
        self.modalities_config = (
            normalize_unique_modalities(modalities)
            if modalities is not None
            else None
        )
        self.raw_modality = normalize_unique_modalities([raw_modality])[0]
        self.context_modalities: tuple[MultimodalModality, ...] = ()
        self.d_model = d_model
        self.encoder_kwargs = dict(encoder_kwargs or {})
        self.fusion_kwargs = dict(fusion_kwargs or {})
        self.aux_output_config = aux_output_config or AuxOutputConfig()
        self.auto_build_on_forward = auto_build_on_forward

        self.encoders: nn.ModuleDict | None = None
        self.fusion: nn.Module | None = None

        hidden_dim = head_hidden_dim or d_model
        self.head = nn.Sequential(
            LinLnDrop(
                d_model,
                hidden_dim,
                ln=True,
                p=head_dropout,
                act=get_activation_fn(head_activation),
            ),
            LinLnDrop(
                hidden_dim,
                num_classes,
                ln=False,
                p=head_dropout,
                act=None,
            ),
        )

    def _build_fusion_module(self, n_modalities: int) -> None:
        n_fusion_inputs = self.fusion_entry.resolve_input_count(n_modalities)
        self.fusion = self.fusion_entry.fusion_class(
            **{self.fusion_entry.inputs_param_name: n_fusion_inputs},
            d_model=self.d_model,
            **self.fusion_kwargs,
        )

    def build(self, bundle: MultimodalDataBundle) -> "ConfigurableMultimodalFusionClassifier":
        """Build encoders and fusion modules from a concrete bundle."""

        bundle = validate_multimodal_bundle_input(bundle)
        return self.build_from_shapes(bundle.shapes, bundle_modalities=bundle.modalities.keys())

    def build_from_shapes(
        self,
        shapes: Mapping[MultimodalModality, tuple[int, ...]],
        *,
        bundle_modalities: Sequence[MultimodalModality] | None = None,
    ) -> "ConfigurableMultimodalFusionClassifier":
        """Build encoders and fusion modules from modality shapes."""

        if self.modalities_config is None:
            if bundle_modalities is None:
                resolved = tuple(shapes)
            else:
                resolved = tuple(bundle_modalities)
            self.modalities = normalize_unique_modalities(resolved)
        else:
            self.modalities = normalize_unique_modalities(self.modalities_config)

        validate_encoder_registry_has_modalities(
            modalities=self.modalities,
            preset_registry=ENCODER_PRESET_BUILDERS,
        )
        if self.fusion_entry.requires_raw:
            validate_context_modalities_for_raw_centered(
                raw_modality=self.raw_modality,
                modalities=self.modalities,
            )
        self.context_modalities = tuple(
            modality for modality in self.modalities if modality != self.raw_modality
        )

        modalities = require_resolved_modalities(self.modalities)
        validate_modalities_presence(
            required_modalities=modalities,
            available_modalities=shapes.keys(),
            source_label="Shape mapping",
        )
        encoders = nn.ModuleDict()
        for modality in modalities:
            preset_entry = ENCODER_PRESET_BUILDERS[modality]
            shape = shapes[modality]
            modality_kwargs = dict(self.encoder_kwargs.get(modality.value, {}))
            config = preset_entry.build_config(
                shape=shape,
                d_model=self.d_model,
                kwargs=modality_kwargs,
            )
            encoders[modality.value] = build_encoder(config)
        self.encoders = encoders
        self._build_fusion_module(n_modalities=len(modalities))
        return self

    def _create_core_aux(
        self,
        logits: torch.Tensor,
        h_final: torch.Tensor,
        fusion_aux: dict[str, Any],
        embeddings: dict[MultimodalModality, torch.Tensor],
    ) -> FusionAuxOutput:
        encoders, fusion, modalities = require_initialized_model_parts(
            encoders=self.encoders,
            fusion=self.fusion,
            modalities=self.modalities,
        )

        return FusionAuxOutput.from_components(
            logits=logits,
            h_final=h_final,
            active_modalities=[modality.value for modality in modalities],
            embedding_dim=self.d_model,
            fusion_aux=fusion_aux,
            include_fusion_aux=self.aux_output_config.include_fusion_aux,
            include_num_parameters=self.aux_output_config.include_num_parameters,
            include_embeddings=self.aux_output_config.include_embeddings,
            num_parameters={
                "total": count_parameters(self),
                "encoders": {
                    modality.value: count_parameters(encoders[modality.value])
                    for modality in modalities
                },
                "fusion": count_parameters(fusion),
                "head": count_parameters(self.head),
            },
            embeddings={
                modality.value: embeddings[modality] for modality in modalities
            },
        )

    def forward(
        self,
        input: MultimodalDataBundle,
        return_aux: bool = False,
    ) -> torch.Tensor | FusionAuxOutput:
        input = validate_multimodal_bundle_input(input)
        if self.encoders is None:
            if not self.auto_build_on_forward:
                raise ValueError(
                    "Model is not built. Call build(bundle) or build_from_shapes(shapes) "
                    "before forward, or set auto_build_on_forward=True."
                )
            self.build(input)
        encoders, fusion, modalities = require_initialized_model_parts(
            encoders=self.encoders,
            fusion=self.fusion,
            modalities=self.modalities,
        )

        inputs = input.modalities
        validate_modalities_presence(
            required_modalities=modalities,
            available_modalities=inputs.keys(),
            source_label="Inputs",
        )

        embeddings: dict[MultimodalModality, torch.Tensor] = {}
        for modality in modalities:
            embeddings[modality] = encoders[modality.value](inputs[modality])

        fusion_output = fusion.fuse(
            embeddings=embeddings,
            modalities=modalities,
            raw_modality=self.raw_modality if self.fusion_entry.requires_raw else None,
            return_aux=return_aux,
        )
        if return_aux:
            fusion_aux = dict(fusion_output)
            h_final = fusion_aux["h_final"]
        else:
            fusion_aux = {}
            h_final = fusion_output

        logits = self.head(h_final)
        if not return_aux:
            return logits

        aux = self._create_core_aux(
            logits=logits,
            h_final=h_final,
            fusion_aux=fusion_aux,
            embeddings=embeddings,
        )
        return aux
