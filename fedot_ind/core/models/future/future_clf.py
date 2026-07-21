"""Composable multimodal classifier for FUTURE fusion methods."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from fedot_ind.core.models.future.enums import FusionMethod
from fedot_ind.core.models.future.mapping import (
    FUSION_REGISTRY,
    RAW_CENTERED_FUSION_METHODS,
)
from fedot_ind.core.models.future.rules import (
    normalize_unique_modalities,
    resolve_modalities_from_bundle,
    validate_context_modalities_for_raw_centered,
    validate_encoder_registry_has_modalities,
    validate_supported_modalities,
    validate_modalities_presence,
    validate_multimodal_bundle_input,
    validate_positive_int,
    validate_supported_fusion_method,
    require_initialized_model_parts,
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

    def _build_fusion_module(self) -> None:
        fusion_class, inputs_param_name = FUSION_REGISTRY[self.fusion_method]
        n_fusion_inputs = (
            len(self.context_modalities)
            if self.fusion_method in RAW_CENTERED_FUSION_METHODS
            else len(self.modalities)
        )
        self.fusion = fusion_class(
            **{inputs_param_name: n_fusion_inputs},
            d_model=self.d_model,
            **self.fusion_kwargs,
        )

    def _resolve_model_modalities(self, bundle: MultimodalDataBundle) -> None:
        self.modalities = resolve_modalities_from_bundle(
            bundle=bundle,
            modalities=self.modalities_config,
        )
        validate_encoder_registry_has_modalities(
            modalities=self.modalities,
            preset_registry=ENCODER_PRESET_BUILDERS,
        )
        validate_supported_modalities(self.modalities)
        if self.fusion_method in RAW_CENTERED_FUSION_METHODS:
            validate_context_modalities_for_raw_centered(
                raw_modality=self.raw_modality,
                modalities=self.modalities,
            )
        self.context_modalities = tuple(
            modality for modality in self.modalities if modality != self.raw_modality
        )

    def _initialize_encoders(self, bundle: MultimodalDataBundle) -> None:
        if self.modalities is None:
            self._resolve_model_modalities(bundle)
        _, _, modalities = require_initialized_model_parts(
            encoders=None,
            fusion=None,
            modalities=self.modalities,
            require_modules=False,
        )
        validate_modalities_presence(
            required_modalities=modalities,
            available_modalities=bundle.modalities.keys(),
            source_label="Bundle",
        )
        encoders = nn.ModuleDict()
        for modality in modalities:
            preset_builder, shape_key = ENCODER_PRESET_BUILDERS[modality]
            shape = bundle.shapes[modality]
            modality_kwargs = dict(self.encoder_kwargs.get(modality.value, {}))
            config = preset_builder(
                d_model=self.d_model,
                **{shape_key: int(shape[1])},
                **modality_kwargs,
            )
            encoders[modality.value] = build_encoder(config)
        self.encoders = encoders
        self._build_fusion_module()

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

        aux = FusionAuxOutput(
            logits=logits,
            h_final=h_final,
            active_modalities=[modality.value for modality in modalities],
            embedding_dim=self.d_model,
        )
        aux.populate_fusion(
            fusion_aux=fusion_aux,
            include_fusion_aux=self.aux_output_config.include_fusion_aux
        )
        aux.populate_profiling(
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
        return aux

    def forward(
        self,
        input: MultimodalDataBundle,
        return_aux: bool = False,
    ) -> torch.Tensor | FusionAuxOutput:
        input = validate_multimodal_bundle_input(input)
        if self.encoders is None:
            self._initialize_encoders(input)
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

        fusion_aux: dict[str, Any] = {}
        if self.fusion_method in RAW_CENTERED_FUSION_METHODS:
            h_raw = embeddings[self.raw_modality]
            context_embeddings = [embeddings[modality] for modality in self.context_modalities]
            if return_aux:
                fusion_output = fusion(h_raw, *context_embeddings, return_aux=True)
                h_final = fusion_output["h_final"]
                fusion_aux = dict(fusion_output)
            else:
                h_final = fusion(h_raw, *context_embeddings, return_aux=False)
        else:
            ordered_embeddings = [embeddings[modality] for modality in modalities]
            if self.fusion_method is FusionMethod.gated:
                if return_aux:
                    h_final, gates = fusion(*ordered_embeddings, return_gates=True)
                    fusion_aux["gates"] = gates
                else:
                    h_final = fusion(*ordered_embeddings, return_gates=False)
            else:
                h_final = fusion(*ordered_embeddings)

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
