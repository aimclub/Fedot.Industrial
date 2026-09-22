"""Translate the data-only Industrial catalog into a FEDOT extension manifest."""

from __future__ import annotations

from dataclasses import fields
from functools import lru_cache
from importlib.resources import files
import json
from typing import Any, Mapping

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.extensions import (
    ArrayBackend,
    ExtensionManifest,
    ExternalModelSpec,
    ExternalTransformSpec,
    ModelCapabilities,
    ModelHyperparamsSchema,
    TransformCapabilities,
)

from fedot_ind.integration.fedot.extensions.catalog import load_industrial_extension_catalog
from fedot_ind.integration.fedot.extensions.contracts import (
    IndustrialOperationDeclaration,
    IndustrialOperationKind,
)
from fedot_ind.integration.fedot.extensions.factories import make_deferred_factory


DEFAULTS_PACKAGE = "fedot_ind.core.repository.data"
DEFAULTS_RESOURCE = "default_operation_params.json"


@lru_cache(maxsize=1)
def build_industrial_extension_manifest() -> ExtensionManifest:
    """Build one immutable manifest from validated packaged resources."""
    catalog = load_industrial_extension_catalog()
    defaults = _load_default_parameters()
    search_parameters = _load_search_parameters()
    models = []
    transforms = []
    for declaration in catalog.operations:
        schema = _hyperparams_schema(declaration, defaults, search_parameters)
        factory = make_deferred_factory(declaration)
        if declaration.kind is IndustrialOperationKind.MODEL:
            models.append(ExternalModelSpec(
                name=declaration.name,
                factory=factory,
                capabilities=ModelCapabilities(
                    tasks=_task_types(declaration),
                    data_types=_data_types(declaration),
                    tags=declaration.tags,
                    supports_multimodal=declaration.supports_multimodal,
                    backend=ArrayBackend[declaration.backend],
                    output_data_type=DataTypesEnum[declaration.output_data_type],
                    requires_target=declaration.requires_target,
                ),
                hyperparams_schema=schema,
                description=declaration.description,
            ))
        else:
            transforms.append(ExternalTransformSpec(
                name=declaration.name,
                factory=factory,
                capabilities=TransformCapabilities(
                    tasks=_task_types(declaration),
                    data_types=_data_types(declaration),
                    output_data_type=DataTypesEnum[declaration.output_data_type],
                    tags=declaration.tags,
                    supports_multimodal=declaration.supports_multimodal,
                    backend=ArrayBackend[declaration.backend],
                    requires_fit=declaration.requires_fit,
                    requires_target=declaration.requires_target,
                ),
                hyperparams_schema=schema,
                description=declaration.description,
            ))
    return ExtensionManifest(
        name=catalog.name,
        version=catalog.version,
        models=tuple(models),
        transforms=tuple(transforms),
        module=__name__,
        description=f"{catalog.description} Catalog SHA256: {catalog.digest}.",
    )


def _load_default_parameters() -> Mapping[str, Mapping[str, Any]]:
    resource = files(DEFAULTS_PACKAGE).joinpath(DEFAULTS_RESOURCE)
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Industrial default operation parameters must be an object.")
    return payload


def _hyperparams_schema(declaration: IndustrialOperationDeclaration,
                        defaults_by_operation: Mapping[str, Mapping[str, Any]],
                        search_parameters: Mapping[str, Mapping[str, Any]]) -> ModelHyperparamsSchema:
    defaults = dict(defaults_by_operation.get(declaration.defaults_key or declaration.name, {}))
    optional = set(defaults)
    optional.update(search_parameters.get(declaration.name, {}))
    optional.update(_pdl_parameter_names(declaration.name, search_parameters))
    optional.difference_update({"model_fit", "model_predict"})
    return ModelHyperparamsSchema(optional=tuple(sorted(optional)), defaults=defaults)


def _pdl_parameter_names(
        operation_name: str,
        search_parameters: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    task_by_operation = {
        "pdl_clf": "classification",
        "pdl_reg": "regression",
    }
    task = task_by_operation.get(operation_name)
    if task is None:
        return set()

    from fedot_ind.core.models.pdl.base_models import available_pdl_base_models
    from fedot_ind.core.models.pdl.config import PairwiseLearningConfig

    names = {field.name for field in fields(PairwiseLearningConfig)}
    names.add("model")
    for model_name in available_pdl_base_models(task):
        names.update(search_parameters.get(model_name, {}))
    return names


def _load_search_parameters() -> Mapping[str, Mapping[str, Any]]:
    try:
        from fedot_ind.core.tuning.search_space import get_industrial_search_space
    except ImportError:
        # Search-space dependencies are optional for manifest discovery.
        return {}
    return get_industrial_search_space()


def _task_types(declaration: IndustrialOperationDeclaration) -> tuple[TaskTypesEnum, ...]:
    return tuple(TaskTypesEnum[task] for task in declaration.tasks)


def _data_types(declaration: IndustrialOperationDeclaration) -> tuple[DataTypesEnum, ...]:
    return tuple(DataTypesEnum[data_type] for data_type in declaration.data_types)


FEDOT_EXTENSION_MANIFEST = build_industrial_extension_manifest()
