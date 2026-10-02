import json
from dataclasses import replace

from hypothesis import given, strategies as st
import pytest

from fedot_ind.core.optimizer.graph_validation import (
    ComputeDevice,
    GraphNodeSpec,
    GraphSpec,
    NodePosition,
    OperationCapabilities,
    OperationKind,
    StructuralRole,
    ValidationIssue,
    ValidationIssueCode,
    ValidationReport,
    ValidationSeverity,
    validate_graph,
)


def _capability(
        name: str,
        *,
        kind: OperationKind = OperationKind.TRANSFORM,
        inputs: tuple[str, ...] = ("tabular",),
        outputs: tuple[str, ...] = ("tabular",),
        tasks: tuple[str, ...] = ("classification",),
        devices: tuple[ComputeDevice, ...] = (ComputeDevice.CPU,),
        serializable: bool = True,
        positions: tuple[NodePosition, ...] = (NodePosition.ANY,),
        min_parents: int = 0,
        max_parents: int | None = None,
        identical: bool = True,
        tags: tuple[str, ...] = (),
        structural_role: StructuralRole | None = None,
) -> OperationCapabilities:
    return OperationCapabilities(
        name=name,
        kind=kind,
        input_data_types=inputs,
        output_data_types=outputs,
        task_types=tasks,
        devices=devices,
        serializable=serializable,
        allowed_positions=positions,
        min_parents=min_parents,
        max_parents=max_parents,
        allows_identical_parents=identical,
        tags=tags,
        structural_role=structural_role,
    )


def _valid_spec() -> GraphSpec:
    return GraphSpec(
        graph_id="valid",
        task_type="classification",
        nodes=(
            GraphNodeSpec("source", "scale", capabilities=_capability("scale")),
            GraphNodeSpec(
                "root",
                "classifier",
                parent_ids=("source",),
                capabilities=_capability("classifier", kind=OperationKind.MODEL),
            ),
        ),
    )


def test_valid_graph_has_empty_serializable_report():
    report = validate_graph(_valid_spec())

    assert report.is_valid
    assert report.error_codes == ()
    assert json.loads(json.dumps(report.to_record())) == report.to_record()


def test_issue_context_is_deeply_immutable_but_serializes_as_json_values():
    source = ["classification", "regression"]
    issue = ValidationIssue.create(
        ValidationIssueCode.TASK_NOT_SUPPORTED,
        "Unsupported task.",
        context={"supported": source, "nested": {"devices": {"cpu", "cuda"}}},
    )
    source.append("ts_forecasting")

    record = issue.to_record()

    assert record["context"]["supported"] == ["classification", "regression"]
    assert set(record["context"]["nested"]["devices"]) == {"cpu", "cuda"}
    hash(issue)


def test_warning_does_not_reject_graph_and_report_append_is_immutable():
    original = ValidationReport("graph")
    warning = ValidationIssue.create(
        ValidationIssueCode.LEGACY_VERIFIER_REJECTED,
        "Compatibility warning.",
        severity=ValidationSeverity.WARNING,
    )

    updated = original.append(warning)

    assert original.issues == ()
    assert updated.is_valid
    assert updated.error_codes == ()


def test_graph_contract_rejects_mutable_collection_inputs():
    with pytest.raises(TypeError, match="collections must be tuples"):
        OperationCapabilities(
            name="invalid",
            kind=OperationKind.MODEL,
            input_data_types=["tabular"],
            output_data_types=("tabular",),
            task_types=("classification",),
        )
    with pytest.raises(TypeError, match="must be a tuple"):
        GraphNodeSpec("node", "operation", parent_ids=[])
    with pytest.raises(TypeError, match="nodes must be a tuple"):
        GraphSpec("graph", nodes=[])


@pytest.mark.parametrize(
    "values",
    [
        {"min_parents": -1},
        {"max_parents": False},
        {"min_parents": 2, "max_parents": 1},
        {"devices": ()},
        {"devices": (ComputeDevice.CPU, ComputeDevice.CPU)},
        {"positions": (NodePosition.ANY, NodePosition.ROOT)},
    ],
)
def test_capability_constructor_rejects_ambiguous_constraints(values):
    with pytest.raises((TypeError, ValueError)):
        _capability("invalid", **values)


def test_validator_collects_independent_structural_failures_without_early_return():
    spec = GraphSpec(
        graph_id="broken-structure",
        nodes=(
            GraphNodeSpec("same", "left", parent_ids=("missing",)),
            GraphNodeSpec("same", "right", parent_ids=("same",)),
            GraphNodeSpec("isolated", "isolated"),
            GraphNodeSpec("cycle-a", "cycle-a", parent_ids=("cycle-b",)),
            GraphNodeSpec("cycle-b", "cycle-b", parent_ids=("cycle-a",)),
        ),
    )

    report = validate_graph(spec)

    assert not report.is_valid
    assert {
        ValidationIssueCode.DUPLICATE_NODE_ID.value,
        ValidationIssueCode.UNKNOWN_PARENT.value,
        ValidationIssueCode.SELF_CYCLE.value,
        ValidationIssueCode.CYCLE.value,
        ValidationIssueCode.ROOT_COUNT.value,
        ValidationIssueCode.ISOLATED_NODE.value,
        ValidationIssueCode.DISCONNECTED_COMPONENT.value,
        ValidationIssueCode.UNKNOWN_OPERATION.value,
    }.issubset(report.error_codes)


def test_capability_rule_reports_every_contract_dimension():
    source = GraphNodeSpec(
        "source",
        "gpu-only-transform",
        capabilities=_capability(
            "gpu-only-transform",
            tasks=("regression",),
            devices=(ComputeDevice.CUDA,),
            serializable=False,
            positions=(NodePosition.SECONDARY,),
            min_parents=1,
        ),
    )
    root = GraphNodeSpec(
        "root",
        "not-a-model",
        parent_ids=("source",),
        capabilities=_capability(
            "not-a-model",
            max_parents=0,
        ),
    )
    spec = GraphSpec(
        graph_id="capability-errors",
        nodes=(source, root),
        task_type="classification",
        requested_device=ComputeDevice.CPU,
        require_serializable=True,
    )

    report = validate_graph(spec)

    assert {
        ValidationIssueCode.POSITION_NOT_ALLOWED.value,
        ValidationIssueCode.TOO_FEW_PARENTS.value,
        ValidationIssueCode.TOO_MANY_PARENTS.value,
        ValidationIssueCode.TASK_NOT_SUPPORTED.value,
        ValidationIssueCode.DEVICE_NOT_SUPPORTED.value,
        ValidationIssueCode.OPERATION_NOT_SERIALIZABLE.value,
        ValidationIssueCode.ROOT_NOT_MODEL.value,
    }.issubset(report.error_codes)


def test_data_flow_reports_each_incompatible_edge_and_duplicate_transform_parent():
    image = _capability("image", outputs=("image",))
    model = _capability(
        "model",
        kind=OperationKind.MODEL,
        inputs=("tabular",),
    )
    spec = GraphSpec(
        graph_id="flow",
        task_type="classification",
        nodes=(
            GraphNodeSpec("a", "image", capabilities=image),
            GraphNodeSpec("b", "image", capabilities=image),
            GraphNodeSpec("root", "model", ("a", "b"), model),
        ),
    )

    report = validate_graph(spec)

    assert report.error_codes.count(ValidationIssueCode.DATA_TYPE_MISMATCH.value) == 2
    assert ValidationIssueCode.IDENTICAL_PARENT_OPERATIONS.value in report.error_codes


def test_class_decompose_allows_regression_model_inside_classification_graph():
    model = _capability("regressor", kind=OperationKind.MODEL, tasks=("regression",))
    decompose = _capability("custom_class_split", structural_role=StructuralRole.CLASS_DECOMPOSITION)
    spec = GraphSpec(
        graph_id="multitask",
        task_type="classification",
        nodes=(
            GraphNodeSpec("left", "left", capabilities=_capability("left")),
            GraphNodeSpec("right", "right", capabilities=_capability("right")),
            GraphNodeSpec("bridge", "custom_class_split", ("left", "right"), decompose),
            GraphNodeSpec("root", "regressor", ("bridge",), model),
        ),
    )

    report = validate_graph(spec)

    root_task_issues = [issue for issue in report.issues
                        if issue.code is ValidationIssueCode.TASK_NOT_SUPPORTED
                        and "root" in issue.node_ids]
    assert root_task_issues == []


def test_special_rules_handle_resample_decompose_and_mixed_sources_safely():
    transform = _capability("transform")
    model = _capability("model", kind=OperationKind.MODEL)
    spec = GraphSpec(
        graph_id="special",
        task_type="classification",
        nodes=(
            GraphNodeSpec("data", "data_source/table", capabilities=_capability(
                "data_source/table", kind=OperationKind.DATA_SOURCE)),
            GraphNodeSpec("resample", "resample", capabilities=_capability(
                "resample", structural_role=StructuralRole.RESAMPLING)),
            GraphNodeSpec("ordinary", "scale", capabilities=transform),
            GraphNodeSpec("decompose", "decompose", ("ordinary",), _capability(
                "decompose", structural_role=StructuralRole.DECOMPOSITION)),
            GraphNodeSpec("root", "model", ("resample", "decompose"), model),
        ),
    )

    report = validate_graph(spec)

    assert {
        ValidationIssueCode.MIXED_DATA_SOURCES.value,
        ValidationIssueCode.RESAMPLE_POSITION.value,
        ValidationIssueCode.RESAMPLE_PARENT_CONFLICT.value,
        ValidationIssueCode.DECOMPOSE_PARENT_COUNT.value,
    }.issubset(report.error_codes)


def test_special_graph_rules_follow_declared_roles_when_operations_are_renamed():
    model = _capability("model", kind=OperationKind.MODEL)
    resampling = _capability("custom_sampler", structural_role=StructuralRole.RESAMPLING)
    decomposition = _capability("custom_split", structural_role=StructuralRole.DECOMPOSITION)
    spec = GraphSpec(
        graph_id="extension-roles", task_type="classification",
        nodes=(
            GraphNodeSpec("sampler", "custom_sampler", capabilities=resampling),
            GraphNodeSpec("ordinary", "scale", capabilities=_capability("scale")),
            GraphNodeSpec("split", "custom_split", ("ordinary",), decomposition),
            GraphNodeSpec("root", "model", ("sampler", "split"), model),
        ),
    )

    codes = validate_graph(spec).error_codes
    assert ValidationIssueCode.RESAMPLE_POSITION.value in codes
    assert ValidationIssueCode.RESAMPLE_PARENT_CONFLICT.value in codes
    assert ValidationIssueCode.DECOMPOSE_PARENT_COUNT.value in codes

    without_roles = replace(spec, nodes=tuple(replace(
        node, capabilities=(None if node.capabilities is None else replace(
            node.capabilities, structural_role=None))) for node in spec.nodes))
    familiar_name = replace(without_roles, nodes=tuple(replace(
        node, operation_name="resample" if node.node_id == "sampler" else node.operation_name)
        for node in without_roles.nodes))
    assert ValidationIssueCode.RESAMPLE_POSITION.value not in validate_graph(familiar_name).error_codes
    assert ValidationIssueCode.DECOMPOSE_PARENT_COUNT.value not in validate_graph(familiar_name).error_codes


def test_parallel_filtering_rule_does_not_reject_sequential_filters():
    filtering = _capability("filter", tags=("filtering",))
    model = _capability("model", kind=OperationKind.MODEL)
    sequential = GraphSpec(
        graph_id="sequential",
        task_type="classification",
        nodes=(
            GraphNodeSpec("first", "filter", capabilities=filtering),
            GraphNodeSpec("second", "filter", ("first",), filtering),
            GraphNodeSpec("root", "model", ("second",), model),
        ),
    )
    parallel = GraphSpec(
        graph_id="parallel",
        task_type="classification",
        nodes=(
            GraphNodeSpec("left", "filter", capabilities=filtering),
            GraphNodeSpec("right", "filter", capabilities=filtering),
            GraphNodeSpec("root", "model", ("left", "right"), model),
        ),
    )

    assert ValidationIssueCode.PARALLEL_FILTERING_BRANCHES.value not in validate_graph(
        sequential).error_codes
    assert ValidationIssueCode.PARALLEL_FILTERING_BRANCHES.value in validate_graph(
        parallel).error_codes


def test_validation_order_does_not_depend_on_node_order():
    spec = _valid_spec()
    reversed_spec = GraphSpec(
        graph_id=spec.graph_id,
        task_type=spec.task_type,
        nodes=tuple(reversed(spec.nodes)),
    )

    assert validate_graph(spec).to_record() == validate_graph(reversed_spec).to_record()


@given(st.lists(st.integers(min_value=0, max_value=50), min_size=1, max_size=20))
def test_forward_parent_graphs_never_report_cycles(raw_parents):
    nodes = []
    transform = _capability("transform")
    for index, raw_parent in enumerate(raw_parents):
        parent_ids = () if index == 0 else (f"node-{raw_parent % index}",)
        nodes.append(GraphNodeSpec(
            f"node-{index}",
            f"operation-{index}",
            parent_ids,
            transform,
        ))
    nodes.append(GraphNodeSpec(
        "root",
        "model",
        (f"node-{len(nodes) - 1}",),
        _capability("model", kind=OperationKind.MODEL),
    ))
    spec = GraphSpec("acyclic", tuple(nodes), task_type="classification")

    assert ValidationIssueCode.CYCLE.value not in validate_graph(spec).error_codes


@given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=12))
def test_self_parent_is_always_reported_without_index_access(name):
    spec = GraphSpec(
        "self-cycle",
        (GraphNodeSpec(name, "operation", (name,), _capability("operation")),),
        task_type="classification",
    )

    codes = validate_graph(spec).error_codes

    assert ValidationIssueCode.SELF_CYCLE.value in codes
    assert ValidationIssueCode.CYCLE.value in codes
