from dataclasses import replace

import pytest
from hypothesis import given, strategies as st

from fedot_ind.core.optimizer.graph_validation import (
    GraphNodeSpec, GraphSpec, OperationCapabilities, OperationKind, validate_graph)
from fedot_ind.core.optimizer.mutation_planning import (
    AppliedMutation, ForkFeatureBranch, MutationKind, MutationLimits, MutationPlan, MutationRejection,
    PruneFeatureBranch, RawInputSpec, RewriteChain, apply_mutation_plan, enumerate_mutation_plans, graph_signature)


def transform(name, source='tabular', output='tabular'):
    return OperationCapabilities(name, OperationKind.TRANSFORM, task_types=('regression',),
                                 input_data_types=(source,), output_data_types=(output,), max_parents=1)


def chain(count=2):
    nodes = tuple(GraphNodeSpec(f't{i}', f't{i}', () if i == 0 else (f't{i-1}',), transform(f't{i}'))
                  for i in range(count))
    model = OperationCapabilities('model', OperationKind.MODEL, task_types=('regression',),
                                  input_data_types=('tabular',), output_data_types=('tabular',))
    return GraphSpec('test', nodes + (GraphNodeSpec('m', 'model', (f't{count-1}',), model),),
                     task_type='regression', input_data_type='tabular')


RAW = RawInputSpec('tabular', True)
LIMITS = MutationLimits(16, 16, max_candidates=128, max_arity=3)


def apply(graph, edit, raw=RAW, limits=LIMITS):
    return apply_mutation_plan(graph, MutationPlan(graph_signature(graph), (edit,), raw), limits=limits, raw_input=raw)


@given(st.integers(1, 3), st.integers(1, 3))
def test_rewrite_is_atomic_valid_and_does_not_mutate_the_source(old_count, new_count):
    original = chain(old_count)
    before = original.to_record()
    edit = RewriteChain(tuple(f't{i}' for i in range(old_count)),
                        tuple(transform(f'new{i}') for i in range(new_count)))
    result = apply(original, edit)
    assert isinstance(result, AppliedMutation)
    assert validate_graph(result.graph).is_valid
    assert len(result.graph.nodes) == new_count + 1
    assert original.to_record() == before


def test_atomic_chain_can_cross_an_invalid_intermediate_representation():
    old = chain()
    old = replace(old, input_data_type='ts', nodes=(
        GraphNodeSpec('t0', 'extract', (), transform('extract', 'ts')),
        old.nodes[1], old.nodes[2]))
    edit = RewriteChain(('t0', 't1'), (transform('image', 'ts', 'image'), transform('flatten', 'image')))
    result = apply(old, edit, RawInputSpec('ts', True))
    assert isinstance(result, AppliedMutation)
    assert validate_graph(result.graph).is_valid


def test_fork_and_whole_branch_pruning_restore_the_original_graph():
    original = chain(1)
    forked = apply(original, ForkFeatureBranch('m', None, (transform('b'), transform('c'))))
    assert isinstance(forked, AppliedMutation)
    root = next(node for node in forked.graph.nodes if node.node_id == 'm')
    assert len(root.parent_ids) == 2 and len(forked.graph.nodes) == 4
    pruned = apply(forked.graph, PruneFeatureBranch('m', root.parent_ids[-1]))
    assert isinstance(pruned, AppliedMutation)
    assert pruned.graph == original


def test_pruning_keeps_a_shared_prefix():
    original = chain(1)
    forked = apply(original, ForkFeatureBranch('m', 't0', (transform('pca'),)))
    assert isinstance(forked, AppliedMutation)
    root = next(node for node in forked.graph.nodes if node.node_id == 'm')
    pruned = apply(forked.graph, PruneFeatureBranch('m', root.parent_ids[-1]))
    assert isinstance(pruned, AppliedMutation) and pruned.graph == original


@pytest.mark.parametrize('edit', [ForkFeatureBranch('m', None, (transform('b'),)),
                                  RewriteChain(('t0',), (transform('b'), transform('c')))])
def test_new_primary_representations_require_confirmed_input(edit):
    assert isinstance(apply(chain(1), edit, None), MutationRejection)


def test_rewrite_cannot_delete_a_shared_internal_node():
    graph = chain()
    root = replace(graph.nodes[-1], parent_ids=('t0', 't1'))
    graph = replace(graph, nodes=(*graph.nodes[:-1], root))
    result = apply(graph, RewriteChain(('t0', 't1'), (transform('other'),)))
    assert isinstance(result, MutationRejection)


def test_multimodal_capability_in_parallel_context_is_not_assumed_safe():
    graph = chain(2)
    extra = GraphNodeSpec('extra', 'extra', (), transform('extra'))
    root = replace(graph.nodes[-1], parent_ids=('t1', 'extra'))
    graph = replace(graph, nodes=(*graph.nodes[:-1], extra, root), input_data_type=None)
    operation = replace(transform('multi'), supports_multimodal=True)
    assert isinstance(apply(graph, RewriteChain(('t1',), (operation,)), None), MutationRejection)


def test_rewrites_cannot_escape_depth_or_arity_limits():
    graph = chain(1)
    edit = ForkFeatureBranch('m', None, (transform('b'), transform('c')))
    assert isinstance(apply(graph, edit, limits=replace(LIMITS, max_arity=1)), MutationRejection)
    assert isinstance(apply(graph, edit, limits=replace(LIMITS, max_depth=2)), MutationRejection)


def test_rewrite_enumeration_is_opt_in_bounded_and_deterministic():
    graph = chain(1)
    catalog = (transform('b'), transform('c'))
    default = enumerate_mutation_plans(graph, catalog, limits=LIMITS, raw_input=RAW)
    assert all(not isinstance(p.edits[0], (RewriteChain, ForkFeatureBranch)) for p in default.plans)
    kwargs = dict(limits=replace(LIMITS, max_candidates=12), raw_input=RAW, kinds=(MutationKind.REWRITE,))
    first = enumerate_mutation_plans(graph, catalog, **kwargs)
    assert first == enumerate_mutation_plans(graph, catalog, **kwargs)
    assert first.examined <= 12 and first.plans
    assert any(len(apply_mutation_plan(graph, p, limits=LIMITS, raw_input=RAW).graph.nodes) > 2 for p in first.plans)
