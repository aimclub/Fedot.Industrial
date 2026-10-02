from types import SimpleNamespace
from dataclasses import replace

import pytest
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.verification import verifier_for_task
from fedot.core.repository.tasks import TaskTypesEnum

from fedot_ind.core.optimizer.graph_validation import (
    ComputeDevice,
    IndustrialGraphVerifier,
    OperationKind,
    StructuralRole,
    ValidationIssueCode,
    capabilities_from_declaration,
    graph_spec_from_runtime,
    industrial_capability_index,
    infer_validation_context,
)
from fedot_ind.integration.fedot.extensions.catalog import load_industrial_extension_catalog


def _classification_pipeline() -> Pipeline:
    return Pipeline(SecondaryNode("rf", nodes_from=[PrimaryNode("scaling")]))


def test_pipeline_adapter_projects_fedot_metadata_without_runtime_objects():
    pipeline = _classification_pipeline()

    spec = graph_spec_from_runtime(
        pipeline,
        task_type=TaskTypesEnum.classification,
    )

    by_operation = {node.operation_name: node for node in spec.nodes}
    assert spec.task_type == "classification"
    assert by_operation["scaling"].capabilities.output_data_types == ("tabular", "ts")
    assert by_operation["rf"].capabilities.kind is OperationKind.MODEL
    assert by_operation["rf"].parent_ids == (by_operation["scaling"].node_id,)


def test_mapping_metadata_and_device_aliases_are_projected_at_the_boundary():
    node = SimpleNamespace(
        content={
            "name": "custom_transform/version",
            "metadata": {
                "input_types": "table",
                "output_types": "time_series",
                "task_type": "classification",
                "tags": "torch",
                "allowed_positions": None,
            },
        },
        nodes_from=[],
    )
    spec = graph_spec_from_runtime(
        SimpleNamespace(nodes=[node]),
        task_type="classification",
        requested_device="gpu",
        require_serializable=True,
        capability_index={},
    )

    capability = spec.nodes[0].capabilities
    assert spec.graph_id == "graph:1"
    assert spec.requested_device is ComputeDevice.CUDA
    assert spec.require_serializable is True
    assert spec.nodes[0].operation_name == "custom_transform"
    assert capability.input_data_types == ("tabular",)
    assert capability.output_data_types == ("ts",)
    assert capability.devices == (ComputeDevice.CPU, ComputeDevice.CUDA)


def test_adapter_translates_legacy_names_and_explicit_extension_roles():
    def node(name, role=None):
        metadata = {"input_types": ["tabular"], "output_types": ["tabular"]}
        if role is not None:
            metadata["structural_role"] = role
        return SimpleNamespace(content={"name": name, "metadata": metadata}, nodes_from=[])

    spec = graph_spec_from_runtime(SimpleNamespace(nodes=[
        node("resample"), node("decompose"), node("class_decompose"),
        node("custom_sampler", "resampling"),
    ]), capability_index={})

    assert [item.capabilities.structural_role for item in spec.nodes] == [
        StructuralRole.RESAMPLING, StructuralRole.DECOMPOSITION,
        StructuralRole.CLASS_DECOMPOSITION, StructuralRole.RESAMPLING,
    ]


def test_runtime_node_without_catalog_or_metadata_remains_explicitly_unknown():
    node = SimpleNamespace(content={"name": "unknown"}, nodes_from=[])

    spec = graph_spec_from_runtime(SimpleNamespace(nodes=[node]), capability_index={})

    assert spec.nodes[0].capabilities is None


def test_runtime_boundary_rejects_invalid_adapter_and_device():
    with pytest.raises(TypeError, match="callable restore"):
        graph_spec_from_runtime(object(), adapter=object())
    with pytest.raises(ValueError, match="Unsupported validation device"):
        graph_spec_from_runtime(SimpleNamespace(nodes=[]), requested_device="tpu")
    with pytest.raises(TypeError, match="require_serializable"):
        graph_spec_from_runtime(SimpleNamespace(nodes=[]), require_serializable="false")


def test_runtime_boundary_rejects_unknown_explicit_node_position():
    node = SimpleNamespace(
        content={
            "name": "custom_transform",
            "metadata": {
                "input_types": ["table"],
                "output_types": ["table"],
                "task_type": ["classification"],
                "allowed_positions": ["middle"],
            },
        },
        nodes_from=[],
    )

    with pytest.raises(ValueError, match="Unsupported operation positions"):
        graph_spec_from_runtime(
            SimpleNamespace(nodes=[node]),
            capability_index={},
        )


def test_opt_graph_is_restored_once_before_specification_building():
    pipeline = _classification_pipeline()
    adapter = PipelineAdapter()
    opt_graph = adapter.adapt(pipeline)
    calls = []

    class RecordingAdapter:
        def restore(self, graph):
            calls.append(graph)
            return adapter.restore(graph)

    spec = graph_spec_from_runtime(
        opt_graph,
        adapter=RecordingAdapter(),
        task_type="classification",
    )

    assert len(calls) == 1
    assert len(spec.nodes) == 2


def test_compatibility_verifier_reuses_the_single_restored_pipeline():
    pipeline = _classification_pipeline()
    adapter = PipelineAdapter()
    opt_graph = adapter.adapt(pipeline)
    restore_calls = []
    legacy_inputs = []

    class RecordingAdapter:
        def restore(self, graph):
            restore_calls.append(graph)
            return adapter.restore(graph)

    verifier = IndustrialGraphVerifier(
        adapter=RecordingAdapter(),
        task_type="classification",
        legacy_verifier=lambda graph: legacy_inputs.append(graph) or True,
    )

    assert verifier(opt_graph) is True
    assert restore_calls == [opt_graph]
    assert len(legacy_inputs) == 1
    assert isinstance(legacy_inputs[0], Pipeline)


def test_industrial_declaration_exposes_backend_devices_and_contract_fields():
    declaration = next(
        operation for operation in load_industrial_extension_catalog().operations
        if operation.name == "quantile_extractor_torch"
    )

    capability = capabilities_from_declaration(declaration)

    assert capability.devices == (ComputeDevice.CPU, ComputeDevice.CUDA)
    assert capability.task_types == declaration.tasks
    assert capability.input_data_types == declaration.data_types
    assert capability.output_data_types == (declaration.output_data_type,)
    assert capabilities_from_declaration(replace(
        declaration, structural_role="resampling")).structural_role is StructuralRole.RESAMPLING


def test_industrial_capability_index_is_cached_and_immutable():
    first = industrial_capability_index()
    second = industrial_capability_index()

    assert first is second
    with pytest.raises(TypeError):
        first["unexpected"] = first["pdl_clf"]


def test_context_is_inferred_from_advisor_without_hardcoded_task():
    params = SimpleNamespace(
        advisor=SimpleNamespace(task=SimpleNamespace(
            task_type=TaskTypesEnum.regression)),
        execution_policy=SimpleNamespace(device="cuda"),
    )

    assert infer_validation_context(params) == ("regression", "cuda")


def test_verifier_accepts_valid_pipeline_and_preserves_legacy_check():
    task = TaskTypesEnum.classification
    verifier = IndustrialGraphVerifier(
        adapter=PipelineAdapter(),
        task_type=task,
        legacy_verifier=verifier_for_task(task),
    )

    report = verifier.verify_with_report(_classification_pipeline())

    assert report.is_valid
    assert verifier.last_report is report
    assert verifier(_classification_pipeline()) is True
    assert verifier.verify(_classification_pipeline()) is True


def test_pure_rejection_skips_legacy_verifier_and_retains_all_reasons():
    calls = []
    verifier = IndustrialGraphVerifier(
        adapter=None,
        task_type="classification",
        legacy_verifier=lambda graph: calls.append(graph) or True,
    )

    report = verifier.verify_with_report(SimpleNamespace(nodes=[], descriptive_id="empty"))

    assert not report.is_valid
    assert ValidationIssueCode.EMPTY_GRAPH.value in report.error_codes
    assert calls == []


def test_legacy_rejection_is_visible_as_a_structured_compatibility_issue():
    verifier = IndustrialGraphVerifier(
        adapter=None,
        task_type="classification",
        legacy_verifier=lambda graph: False,
    )

    report = verifier.verify_with_report(_classification_pipeline())

    assert report.error_codes == (ValidationIssueCode.LEGACY_VERIFIER_REJECTED.value,)


def test_expected_legacy_failure_is_visible_without_raising():
    def fail(graph):
        raise ValueError(f"cannot validate {graph}")

    verifier = IndustrialGraphVerifier(
        adapter=None,
        task_type="classification",
        legacy_verifier=fail,
    )

    report = verifier.verify_with_report(_classification_pipeline())

    assert report.error_codes == (ValidationIssueCode.LEGACY_VERIFIER_FAILED.value,)


def test_expected_adaptation_failure_becomes_report_but_os_error_propagates():
    class InvalidAdapter:
        def restore(self, graph):
            raise ValueError("invalid content")

    verifier = IndustrialGraphVerifier(adapter=InvalidAdapter(), task_type=None)
    report = verifier.verify_with_report(object())
    assert report.error_codes == (ValidationIssueCode.GRAPH_ADAPTATION_FAILED.value,)

    class BrokenStorageAdapter:
        def restore(self, graph):
            raise OSError("storage unavailable")

    with pytest.raises(OSError, match="storage unavailable"):
        IndustrialGraphVerifier(
            adapter=BrokenStorageAdapter(), task_type=None).verify_with_report(object())
