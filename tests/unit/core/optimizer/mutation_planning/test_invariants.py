from dataclasses import replace

from hypothesis import given, settings, strategies as st

from fedot_ind.core.optimizer.graph_validation import (
    ComputeDevice, GraphNodeSpec, GraphSpec, NodePosition, OperationKind,
)
from fedot_ind.core.optimizer.mutation_planning import (
    AppliedMutation, MutationLimits, MutationRejection, RepairedGraph,
    admissible_operations, apply_mutation_plan, enumerate_mutation_plans,
    graph_depth, repair_graph, select_mutation_plan,
)
from .test_planning import capability, chain


def independent_graph_oracle(spec, limits):
    """Small DFS reference, deliberately independent of the topological validator."""
    nodes = {node.node_id: node for node in spec.nodes}
    assert len(nodes) == len(spec.nodes) and 0 < len(nodes) <= limits.max_nodes
    referenced = {parent for node in nodes.values() for parent in node.parent_ids}
    roots = set(nodes) - referenced
    assert len(roots) == 1
    root = nodes[next(iter(roots))]
    assert root.capabilities.kind is OperationKind.MODEL

    def visit(node_id, ancestors):
        assert node_id in nodes and node_id not in ancestors
        node = nodes[node_id]
        cap = node.capabilities
        assert node.operation_name == cap.name
        assert spec.task_type in cap.task_types
        assert len(set(node.parent_ids)) == len(node.parent_ids)
        assert cap.min_parents <= len(node.parent_ids)
        assert cap.max_parents is None or len(node.parent_ids) <= cap.max_parents
        assert limits.max_arity is None or len(node.parent_ids) <= limits.max_arity
        position = NodePosition.PRIMARY if not node.parent_ids else NodePosition.SECONDARY
        assert (NodePosition.ANY in cap.allowed_positions or position in cap.allowed_positions
                or (node_id in roots and NodePosition.ROOT in cap.allowed_positions))
        depths, visited = [], {node_id}
        for parent_id in node.parent_ids:
            parent = nodes[parent_id]
            assert set(parent.capabilities.output_data_types) & set(cap.input_data_types)
            depth, seen = visit(parent_id, ancestors | {node_id})
            depths.append(depth)
            visited |= seen
        return 1 + max(depths, default=0), visited

    depth, visited = visit(root.node_id, set())
    assert visited == set(nodes) and depth <= limits.max_depth
    return depth


@st.composite
def graph_and_catalog(draw):
    task = draw(st.sampled_from(("classification", "regression")))
    count = draw(st.integers(0, 6))
    types = draw(st.lists(st.sampled_from(("tabular", "ts", "image")),
                          min_size=count + 1, max_size=count + 1))
    nodes, catalog = [], []
    for index in range(count):
        op = capability(f"transform-{index}", input_data_types=(types[index],),
                        output_data_types=(types[index + 1],), task_types=(task,))
        parents = () if index == 0 else (f"node-{index - 1}",)
        nodes.append(GraphNodeSpec(f"node-{index}", op.name, parents, op))
        catalog.append(replace(op, name=f"alternative-{index}"))
        extra_input = draw(st.sampled_from(("tabular", "ts", "image")))
        extra_output = draw(st.sampled_from(("tabular", "ts", "image")))
        catalog.append(replace(
            op, name=f"union-{index}",
            input_data_types=tuple(sorted(set(op.input_data_types) | {extra_input})),
            output_data_types=tuple(sorted(set(op.output_data_types) | {extra_output}))))
        catalog.append(capability(f"identity-{index}", input_data_types=(types[index],),
                                  output_data_types=(types[index],), task_types=(task,)))
    model = capability("model", OperationKind.MODEL, input_data_types=(types[-1],),
                       task_types=(task,))
    nodes.append(GraphNodeSpec(f"node-{count}", model.name,
                               (f"node-{count - 1}",) if count else (), model))
    catalog.append(replace(model, name="alternative-model"))
    catalog.append(capability("identity", input_data_types=(types[-1],),
                              output_data_types=(types[-1],), task_types=(task,)))
    catalog.append(capability("wrong-task", OperationKind.MODEL,
                              task_types=("regression" if task == "classification" else "classification",)))
    return GraphSpec("generated", tuple(nodes), task_type=task), tuple(catalog)


@settings(max_examples=60, deadline=None)
@given(graph_and_catalog(), st.integers(0, 2**32 - 1), st.integers(0, 30))
def test_plans_are_valid_unique_bounded_and_deterministic(case, seed, work_cap):
    spec, catalog = case
    limits = MutationLimits(len(spec.nodes) + 1, len(spec.nodes) + 1,
                            max_candidates=work_cap, max_arity=1)
    independent_graph_oracle(spec, limits)
    snapshot = spec.to_record()
    candidates = enumerate_mutation_plans(spec, catalog, limits=limits)
    assert candidates == enumerate_mutation_plans(spec, catalog, limits=limits)
    assert candidates.rejection is None and candidates.examined <= work_cap
    assert select_mutation_plan(candidates, seed) == select_mutation_plan(candidates, seed)
    records = set()
    for plan in candidates.plans:
        outcome = apply_mutation_plan(spec, plan, limits=limits)
        assert isinstance(outcome, AppliedMutation)
        assert outcome.graph is not spec
        assert independent_graph_oracle(outcome.graph, limits) == graph_depth(outcome.graph)
        # Operation/edge records, not production graph_signature, detect duplicates.
        record = tuple(sorted((node.node_id, node.operation_name, node.parent_ids)
                              for node in outcome.graph.nodes))
        assert record not in records
        records.add(record)
    assert spec.to_record() == snapshot


type_sets = st.sets(st.sampled_from(("tabular", "ts", "image")), min_size=1).map(
    lambda values: tuple(sorted(values)))


@st.composite
def operation_catalog(draw):
    count = draw(st.integers(0, 15))
    operations = []
    for index in range(count):
        operations.append(capability(
            f"candidate-{index}", draw(st.sampled_from(tuple(OperationKind))),
            input_data_types=draw(type_sets), output_data_types=draw(type_sets),
            task_types=draw(st.sampled_from((("classification",), ("regression",),
                                             ("classification", "regression"), ()))),
            devices=draw(st.sampled_from(((ComputeDevice.CPU,), (ComputeDevice.CUDA,)))),
            serializable=draw(st.booleans()),
            allowed_positions=(draw(st.sampled_from(tuple(NodePosition))),),
            min_parents=draw(st.integers(0, 1)),
        ))
    return tuple(operations)


@settings(max_examples=80, deadline=None)
@given(operation_catalog(), st.integers(0, 15))
def test_admissible_replacements_match_independent_one_node_oracle(catalog, cap):
    spec = replace(chain(0), requested_device=ComputeDevice.CPU, require_serializable=True)
    limits = MutationLimits(1, 1, max_candidates=cap)
    expected = tuple(op for op in catalog[:cap]
                     if op.kind is OperationKind.MODEL
                     and "classification" in op.task_types
                     and "tabular" in op.input_data_types
                     and "tabular" in op.output_data_types
                     and ComputeDevice.CPU in op.devices and op.serializable
                     and op.min_parents == 0
                     and any(pos in op.allowed_positions
                             for pos in (NodePosition.ANY, NodePosition.PRIMARY, NodePosition.ROOT)))
    assert admissible_operations(spec, "node-0", catalog, limits=limits) == expected


@settings(max_examples=60, deadline=None)
@given(st.integers(1, 9), st.integers(0, 9), st.integers(0, 9), st.integers(0, 10))
def test_repair_budget_success_matches_independent_chain_oracle(count, raw_remove, edit_cap, work_cap):
    spec = chain(count)
    remove = raw_remove % count
    limits = MutationLimits(count + 1 - remove, count + 1 - remove,
                            max_candidates=work_cap, max_edits=edit_cap, max_arity=1)
    snapshot = spec.to_record()
    result = repair_graph(spec, limits=limits)
    if remove <= min(edit_cap, work_cap):
        assert isinstance(result, RepairedGraph)
        assert len(result.edits) == result.examined == remove
        assert len(result.graph.nodes) == count + 1 - remove
        independent_graph_oracle(result.graph, limits)
        again = repair_graph(result.graph, limits=limits)
        assert again.graph == result.graph and again.edits == ()
    else:
        assert isinstance(result, MutationRejection)
    assert result.examined <= work_cap
    assert spec.to_record() == snapshot


@settings(max_examples=50, deadline=None)
@given(st.integers(2, 8), st.integers(0, 8))
def test_generated_cycles_are_never_repaired(count, coordinate):
    spec = chain(count)
    index = coordinate % count
    node = replace(spec.nodes[index], parent_ids=(spec.nodes[-1].node_id,))
    spec = replace(spec, nodes=spec.nodes[:index] + (node,) + spec.nodes[index + 1:])
    limits = MutationLimits(12, 12, max_candidates=10, max_edits=4)
    assert isinstance(repair_graph(spec, limits=limits), MutationRejection)
    result = enumerate_mutation_plans(spec, (), limits=limits)
    assert result.rejection is not None and result.examined == 0


@settings(max_examples=50, deadline=None)
@given(st.integers(2, 8), st.integers(0, 8), st.integers(0, 8))
def test_generated_dag_merges_have_reference_depth(count, left, right):
    spec = chain(count)
    root = replace(spec.nodes[-1], parent_ids=tuple(dict.fromkeys(
        (f"node-{count - 1}", f"node-{left % count}", f"node-{right % count}"))),
        capabilities=replace(spec.nodes[-1].capabilities, max_parents=3))
    spec = replace(spec, nodes=spec.nodes[:-1] + (root,))
    limits = MutationLimits(10, 10, max_arity=3)
    assert graph_depth(spec) == independent_graph_oracle(spec, limits) == count + 1
    repaired = repair_graph(spec, limits=limits)
    assert isinstance(repaired, RepairedGraph) and repaired.edits == ()
