from __future__ import annotations

import logging
from typing import Any, Callable

from fedot_ind.core.kernel_learning.generators.base import DEFAULT_GENERATOR_NAMES, OperationSpec
from fedot_ind.core.kernel_learning.generators.lightweight import (
    IdentityFeatureGenerator,
    RandomProjectionEmbeddingFeatureGenerator,
    ShapeletFeatureGenerator,
)
from fedot_ind.core.kernel_learning.generators.repository import (
    BudgetedRepositoryFeatureGeneratorAdapter,
    GeneratorBudgetPolicy,
    PipelineFeatureGeneratorAdapter,
    RepositoryFeatureGeneratorAdapter,
    SummaryFeatureGenerator,
)
from fedot_ind.core.kernel_learning.generators.specs import (
    BASIS_ONLY_GENERATORS,
    _DEFAULT_STATISTICAL_PARAMS,
    eigen_spec,
    fourier_spec,
    recurrence_spec,
    tabular_spec,
    topological_spec,
    torch_quantile_spec,
    wavelet_spec,
)

logger = logging.getLogger(__name__)


def build_generator_registry() -> dict[str, Callable[[], Any]]:
    registry: dict[str, Callable[[], Any]] = {
        "identity": IdentityFeatureGenerator,
        "statistical_summary": SummaryFeatureGenerator,
        "statistical": lambda: SummaryFeatureGenerator(name="statistical"),
        "shapelet_extractor": ShapeletFeatureGenerator,
        "local_pattern_extractor": lambda: ShapeletFeatureGenerator(
            name="local_pattern_extractor",
            n_shapelets=12,
            window_size=5,
        ),
        "embedding_extractor": RandomProjectionEmbeddingFeatureGenerator,
        "foundation_embedding": lambda: RandomProjectionEmbeddingFeatureGenerator(
            name="foundation_embedding",
            n_components=32,
            source="foundation_embedding_adapter",
        ),
        "quantile_extractor": lambda: RepositoryFeatureGeneratorAdapter(
            name="quantile_extractor",
            operation_specs=(torch_quantile_spec(_DEFAULT_STATISTICAL_PARAMS),),
        ),
        "quantile_extractor_torch": lambda: RepositoryFeatureGeneratorAdapter(
            name="quantile_extractor_torch",
            operation_specs=(torch_quantile_spec(_DEFAULT_STATISTICAL_PARAMS),),
        ),
        "wavelet_basis": lambda: RepositoryFeatureGeneratorAdapter(
            name="wavelet_basis",
            operation_specs=(wavelet_spec(),),
        ),
        "wavelet_extractor": lambda: RepositoryFeatureGeneratorAdapter(
            name="wavelet_extractor",
            operation_specs=(wavelet_spec(), torch_quantile_spec(_DEFAULT_STATISTICAL_PARAMS)),
        ),
        "fourier_basis": lambda: RepositoryFeatureGeneratorAdapter(
            name="fourier_basis",
            operation_specs=(fourier_spec(),),
        ),
        "fourier_extractor": lambda: RepositoryFeatureGeneratorAdapter(
            name="fourier_extractor",
            operation_specs=(fourier_spec(), torch_quantile_spec(_DEFAULT_STATISTICAL_PARAMS)),
        ),
        "eigen_basis": lambda: RepositoryFeatureGeneratorAdapter(
            name="eigen_basis",
            operation_specs=(eigen_spec(),),
        ),
        "eigen_extractor": lambda: RepositoryFeatureGeneratorAdapter(
            name="eigen_extractor",
            operation_specs=(eigen_spec(), torch_quantile_spec(_DEFAULT_STATISTICAL_PARAMS)),
        ),
        "recurrence_extractor": lambda: RepositoryFeatureGeneratorAdapter(
            name="recurrence_extractor",
            operation_specs=(recurrence_spec(),),
        ),
        "topological_extractor": lambda: BudgetedRepositoryFeatureGeneratorAdapter(
            name="topological_extractor",
            operation_specs=(topological_spec(),),
            budget_policy=GeneratorBudgetPolicy(
                max_cells=250_000,
                fallback_generator="identity",
            ),
        ),
        "tabular_extractor": lambda: BudgetedRepositoryFeatureGeneratorAdapter(
            name="tabular_extractor",
            operation_specs=(tabular_spec(),),
            budget_policy=GeneratorBudgetPolicy(
                max_samples=25,
                max_cells=10_000,
                fallback_generator="statistical_summary",
            ),
        ),
    }
    extend_with_legacy_pipeline_generators(registry)
    return registry


def create_feature_generator(
        name: str,
        *,
        registry: dict[str, Callable[[], Any]] | None = None,
        torch_device: Any = "auto",
):
    source = registry or build_generator_registry()
    if name not in source:
        raise ValueError(f"Unknown kernel feature generator: {name}")
    generator = source[name]()
    if isinstance(generator, RepositoryFeatureGeneratorAdapter):
        generator.torch_device = torch_device
    return generator


def resolve_generator_operation_specs(
        name: str,
        *,
        materialize_basis: bool = False,
) -> tuple[OperationSpec, ...]:
    generator = create_feature_generator(name)
    if isinstance(generator, IdentityFeatureGenerator):
        return ()
    if not isinstance(generator, RepositoryFeatureGeneratorAdapter):
        return ()

    specs = tuple(generator.operation_specs)
    if (
            materialize_basis
            and len(specs) == 1
            and specs[0].name in BASIS_ONLY_GENERATORS
    ):
        return specs + (torch_quantile_spec(_DEFAULT_STATISTICAL_PARAMS),)
    return specs


def extend_with_legacy_pipeline_generators(registry: dict[str, Callable[[], Any]]) -> None:
    try:
        from fedot_ind.core.repository.constanst_repository import KERNEL_BASELINE_FEATURE_GENERATORS

        for name, pipeline_factory in KERNEL_BASELINE_FEATURE_GENERATORS.items():
            registry.setdefault(
                name,
                lambda generator_name=name, factory=pipeline_factory: PipelineFeatureGeneratorAdapter(
                    name=generator_name,
                    pipeline_factory=factory,
                ),
            )
    except Exception:
        # Keep registry construction importable without the optional FEDOT/dask stack.
        logger.exception("Skipping legacy pipeline generators because optional registry imports failed.")


_extend_with_legacy_pipeline_generators = extend_with_legacy_pipeline_generators


__all__ = [
    "DEFAULT_GENERATOR_NAMES",
    "build_generator_registry",
    "create_feature_generator",
    "extend_with_legacy_pipeline_generators",
    "resolve_generator_operation_specs",
]