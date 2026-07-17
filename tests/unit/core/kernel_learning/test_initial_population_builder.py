import pytest

from fedot_ind.core.kernel_learning.contracts import KernelConfigValidationError, KernelSelectionReport
from fedot_ind.core.kernel_learning.integration import (
    KernelInitialPopulationError,
    KernelInitialPopulationBuilder,
    narrow_kernel_learning_search_space,
    resolve_warm_start_task,
    task_head_candidates,
)
from fedot_ind.core.kernel_learning.selection import KernelImportanceConfig, select_significant_generators

pytest.importorskip("fedot.core.pipelines.pipeline_builder")


def _importance(names, weights, threshold=0.05):
    report = KernelSelectionReport(
        generator_names=tuple(names),
        weights=tuple(weights),
        selected_generators=tuple(names),
        selected_weights=tuple(weights),
        scores={name: float(weight) for name, weight in zip(names, weights)},
        alignments={name: float(weight) for name, weight in zip(names, weights)},
        complexities={name: 0.0 for name in names},
        redundancies={name: 0.0 for name in names},
        task_type="classification",
    )
    return select_significant_generators(report, KernelImportanceConfig(weight_threshold=threshold))


def test_initial_population_builder_creates_single_and_union_specs():
    builder = KernelInitialPopulationBuilder(task_type="classification", head_model="rf")

    specs = builder.build_specs(_importance(("wavelet_extractor", "fourier_extractor"), (0.7, 0.2)))

    assert [spec.kind for spec in specs] == ["single", "single", "union"]
    assert specs[0].generator_names == ("wavelet_extractor",)
    assert specs[-1].generator_names == ("wavelet_extractor", "fourier_extractor")
    assert builder.build_pipeline_from_spec(specs[-1]) is not None


def test_initial_population_builder_can_return_lazy_pipeline_builders():
    builder = KernelInitialPopulationBuilder(task_type="classification", head_model="rf")

    specs = builder.build_specs(_importance(("wavelet_basis",), (1.0,)))
    lazy_builder = builder.build_pipeline_builder_from_spec(specs[0])

    assert lazy_builder.__class__.__name__ == "PipelineBuilder"
    assert callable(getattr(lazy_builder, "build"))


def test_lazy_pipeline_builder_resolves_basis_as_data_operation_after_industrial_repo_init():
    from fedot.core.operations.data_operation import DataOperation
    from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

    builder = KernelInitialPopulationBuilder(task_type="classification", head_model="rf")
    specs = builder.build_specs(_importance(("wavelet_basis",), (1.0,)))
    lazy_builder = builder.build_pipeline_builder_from_spec(specs[0])

    repository = IndustrialModels()
    repository.setup_repository()
    try:
        pipeline = lazy_builder.build()
    finally:
        repository.setup_default_repository()

    operations_by_name = {node.name: node.operation for node in pipeline.nodes}
    assert isinstance(operations_by_name["wavelet_basis"], DataOperation)
    assert isinstance(operations_by_name["quantile_extractor_torch"], DataOperation)


def test_initial_population_builder_materializes_basis_only_generators_to_tabular_features():
    builder = KernelInitialPopulationBuilder(task_type="classification", head_model="rf")

    specs = builder.build_specs(_importance(("wavelet_basis",), (1.0,)))

    assert specs[0].operation_names == ("wavelet_basis", "quantile_extractor_torch", "rf")


def test_initial_population_builder_supports_forecasting_task_heads():
    builder = KernelInitialPopulationBuilder(task_type="ts_forecasting")

    specs = builder.build_specs(_importance(("wavelet_basis",), (1.0,)))

    assert builder.task_type == "forecasting"
    assert builder.resolved_head_model == "ridge"
    assert specs[0].operation_names == ("wavelet_basis", "quantile_extractor_torch", "ridge")


def test_warm_start_task_registry_resolves_aliases_and_heads():
    spec = resolve_warm_start_task("ts_forecasting")

    assert spec.task_type == "forecasting"
    assert spec.default_head_model == "ridge"
    assert task_head_candidates("classification")[0] == "rf"

    with pytest.raises(KernelConfigValidationError, match="Unsupported kernel warm-start task_type"):
        resolve_warm_start_task("vision")


def test_initial_population_builder_skips_identity_without_explicit_fedot_mapping():
    builder = KernelInitialPopulationBuilder(task_type="classification", head_model="rf")

    specs = builder.build_specs(_importance(("identity", "wavelet_basis"), (0.9, 0.8)))

    assert len(specs) == 1
    assert specs[0].generator_names == ("wavelet_basis",)
    assert builder.diagnostics_["skipped_generators"] == {"identity": "no_fedot_operation_chain"}


def test_initial_population_builder_empty_specs_policy_is_explicit():
    builder = KernelInitialPopulationBuilder(task_type="classification", head_model="rf")

    with pytest.raises(KernelInitialPopulationError, match="initial population is empty"):
        builder.build_specs(_importance(("identity",), (1.0,)))

    assert builder.last_specs_ == ()
    assert builder.diagnostics_["empty_specs_reason"] == "all_selected_generators_missing_fedot_operation_chain"

    permissive = KernelInitialPopulationBuilder(
        task_type="classification",
        head_model="rf",
        allow_empty_specs=True,
    )

    assert permissive.build_specs(_importance(("identity",), (1.0,))) == ()
    assert permissive.diagnostics_["empty_specs_reason"] == "all_selected_generators_missing_fedot_operation_chain"


def test_initial_population_builder_limits_union_size():
    builder = KernelInitialPopulationBuilder(task_type="classification", head_model="rf", max_union_size=2)

    specs = builder.build_specs(
        _importance(("wavelet_basis", "fourier_basis", "eigen_basis"), (0.9, 0.8, 0.7))
    )

    union_specs = [spec for spec in specs if spec.kind == "union"]
    assert union_specs[0].generator_names == ("wavelet_basis", "fourier_basis")


def test_narrow_kernel_learning_search_space_keeps_selected_ops_and_excludes_topology():
    builder = KernelInitialPopulationBuilder(task_type="classification", head_model="rf")
    specs = builder.build_specs(_importance(("wavelet_basis",), (1.0,)))

    narrowed = narrow_kernel_learning_search_space(
        available_operations=["rf", "logit", "topological_extractor", "scaling", "normalization"],
        specs=specs,
        task_type="classification",
        head_model="rf",
    )

    assert narrowed == ["wavelet_basis", "quantile_extractor_torch", "rf", "logit", "scaling", "normalization"]
    assert "topological_extractor" not in narrowed
