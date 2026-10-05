"""Public multimodal API with optional implementations loaded on demand."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fedot_ind.core.multimodal.batching import (  # noqa: F401
        MultimodalBundleIndexDataset as MultimodalBundleIndexDataset,
        collate_bundle_indices as collate_bundle_indices,
        iter_bundle_batches as iter_bundle_batches,
        make_bundle_dataloader as make_bundle_dataloader,
        select_bundle_indices as select_bundle_indices,
        split_bundle_by_fraction as split_bundle_by_fraction,
    )
    from fedot_ind.core.multimodal.configs import (  # noqa: F401
        DEFAULT_STAT_FEATURES as DEFAULT_STAT_FEATURES,
        PreparationConfig as PreparationConfig,
        build_preparation_config as build_preparation_config,
        default_normalization_config as default_normalization_config,
        default_transformation_config as default_transformation_config,
    )
    from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle as MultimodalDataBundle  # noqa: F401
    from fedot_ind.core.multimodal.enums import (  # noqa: F401
        MultimodalModality as MultimodalModality,
        NormalizationStep as NormalizationStep,
    )
    from fedot_ind.core.multimodal.mapping import (  # noqa: F401
        MODALITY_CAPABILITIES as MODALITY_CAPABILITIES,
        ModalityCapability as ModalityCapability,
    )
    from fedot_ind.core.multimodal.preparation import MultimodalDatasetPreparer as MultimodalDatasetPreparer  # noqa: F401
    from fedot_ind.core.multimodal.preprocessor import MultimodalPreprocessor as MultimodalPreprocessor  # noqa: F401
    from fedot_ind.core.multimodal.rules import ModalitySpec as ModalitySpec  # noqa: F401
    from fedot_ind.core.operation.transformation.torch_backend.enums import StatisticalFeature as StatisticalFeature  # noqa: F401


_PUBLIC_IMPORTS = {
    "DEFAULT_STAT_FEATURES": ("fedot_ind.core.multimodal.configs", "DEFAULT_STAT_FEATURES"),
    "MODALITY_CAPABILITIES": ("fedot_ind.core.multimodal.mapping", "MODALITY_CAPABILITIES"),
    "ModalitySpec": ("fedot_ind.core.multimodal.rules", "ModalitySpec"),
    "ModalityCapability": ("fedot_ind.core.multimodal.mapping", "ModalityCapability"),
    "MultimodalBundleIndexDataset": (
        "fedot_ind.core.multimodal.batching",
        "MultimodalBundleIndexDataset",
    ),
    "MultimodalDataBundle": ("fedot_ind.core.multimodal.data_bundle", "MultimodalDataBundle"),
    "MultimodalDatasetPreparer": (
        "fedot_ind.core.multimodal.preparation",
        "MultimodalDatasetPreparer",
    ),
    "MultimodalModality": ("fedot_ind.core.multimodal.enums", "MultimodalModality"),
    "MultimodalPreprocessor": (
        "fedot_ind.core.multimodal.preprocessor",
        "MultimodalPreprocessor",
    ),
    "NormalizationStep": ("fedot_ind.core.multimodal.enums", "NormalizationStep"),
    "PreparationConfig": ("fedot_ind.core.multimodal.configs", "PreparationConfig"),
    "StatisticalFeature": (
        "fedot_ind.core.operation.transformation.torch_backend.enums",
        "StatisticalFeature",
    ),
    "build_preparation_config": (
        "fedot_ind.core.multimodal.configs",
        "build_preparation_config",
    ),
    "collate_bundle_indices": (
        "fedot_ind.core.multimodal.batching",
        "collate_bundle_indices",
    ),
    "default_normalization_config": (
        "fedot_ind.core.multimodal.configs",
        "default_normalization_config",
    ),
    "default_transformation_config": (
        "fedot_ind.core.multimodal.configs",
        "default_transformation_config",
    ),
    "iter_bundle_batches": ("fedot_ind.core.multimodal.batching", "iter_bundle_batches"),
    "make_bundle_dataloader": (
        "fedot_ind.core.multimodal.batching",
        "make_bundle_dataloader",
    ),
    "select_bundle_indices": (
        "fedot_ind.core.multimodal.batching",
        "select_bundle_indices",
    ),
    "split_bundle_by_fraction": (
        "fedot_ind.core.multimodal.batching",
        "split_bundle_by_fraction",
    ),
}

__all__ = list(_PUBLIC_IMPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute_name = _PUBLIC_IMPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
