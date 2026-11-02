from dataclasses import replace

import pytest

from fedot_ind.core.optimizer.graph_validation import (
    GraphNodeSpec, GraphSpec, OperationCapabilities, OperationKind, ModelFeatureFlow, validate_graph)
from fedot_ind.core.optimizer.graph_validation.capabilities import industrial_capability_index
from fedot_ind.core.optimizer.mutation_planning import (
    AppliedMutation, ForkForecastBranch, MutationLimits, MutationPlan, MutationRejection,
    RawInputSpec, apply_mutation_plan, graph_signature, enumerate_mutation_plans, MutationKind)


RAW = RawInputSpec('ts', True)
LIMITS = MutationLimits(24, 8, max_arity=4, max_candidates=128)


def initial():
    caps = industrial_capability_index()
    ridge = OperationCapabilities('ridge', OperationKind.MODEL, ('tabular',), ('tabular',), ('ts_forecasting',),
                                  preserves_input_features=True)
    return GraphSpec('forecast', (GraphNodeSpec('h', 'hankelisation', (), caps['hankelisation']),
                                  GraphNodeSpec('m', 'ridge', ('h',), ridge)),
                     task_type='ts_forecasting', input_data_type='ts', model_feature_flow=ModelFeatureFlow.TENSOR_FEATURES)


def fork(graph, child, representation, merge, limits=LIMITS, raw=RAW):
    caps = industrial_capability_index()
    sequence = (caps[representation],)
    if merge == 'forecast_mean':
        sequence += (initial().nodes[1].capabilities,)
    edit = ForkForecastBranch(child, 'h', sequence, caps[merge])
    return apply_mutation_plan(graph, MutationPlan(graph_signature(graph), (edit,), raw), limits=limits, raw_input=raw)


@pytest.mark.parametrize('merge', ['forecast_feature_join', 'forecast_mean'])
def test_two_three_four_branches_are_bounded_valid_and_share_windows(merge):
    graph = initial()
    source = graph
    before = graph.to_record()
    for count, name in enumerate(['forecast_window_center', 'forecast_window_difference', 'forecast_window_smooth'], 2):
        child = next((n.node_id for n in graph.nodes if n.operation_name == merge), 'm')
        result = fork(graph, child, name, merge)
        assert isinstance(result, AppliedMutation), result
        graph = result.graph
        joined = next(n for n in graph.nodes if n.operation_name == merge)
        assert len(joined.parent_ids) == count
        assert validate_graph(graph).is_valid
    assert source.to_record() == before
    child = next(n.node_id for n in graph.nodes if n.operation_name == merge)
    assert isinstance(fork(graph, child, 'forecast_window_center', merge), MutationRejection)


@pytest.mark.parametrize('merge', ['forecast_feature_join', 'forecast_mean'])
def test_forecast_forks_require_confirmed_context_and_respect_budgets(merge):
    assert isinstance(fork(initial(), 'm', 'forecast_window_center', merge, raw=None), MutationRejection)
    assert isinstance(fork(initial(), 'm', 'forecast_window_center', merge,
                           limits=replace(LIMITS, max_arity=1)), MutationRejection)
    assert isinstance(fork(initial(), 'm', 'forecast_window_center', merge,
                           limits=replace(LIMITS, max_nodes=2)), MutationRejection)


def test_enumerator_proposes_both_explicit_merge_kinds():
    caps = industrial_capability_index()
    enabled = tuple(caps[n] for n in ('forecast_feature_join', 'forecast_mean', 'forecast_window_center'))
    enabled += (initial().nodes[1].capabilities,)
    result = enumerate_mutation_plans(initial(), enabled, limits=LIMITS, kinds=(MutationKind.REWRITE,), raw_input=RAW)
    forks = [p.edits[0] for p in result.plans if isinstance(p.edits[0], ForkForecastBranch)]
    assert {e.merge_operation.name for e in forks} == {'forecast_mean', 'forecast_feature_join'}


@pytest.mark.parametrize('parent', ['wavelet_basis', 'quantile_extractor'])
def test_basis_of_coefficients_or_feature_columns_is_rejected(parent):
    caps = industrial_capability_index()
    graph = GraphSpec('basis', (GraphNodeSpec('a', parent, (), caps[parent]),
                                GraphNodeSpec('b', 'fourier_basis', ('a',), caps['fourier_basis'])),
                      task_type='classification', input_data_type='ts')
    assert 'representation_mismatch' in validate_graph(graph).error_codes


@pytest.mark.parametrize('params, allowed', [({'output_format': 'signal'}, True),
                                             ({'output_format': 'spectrum'}, False),
                                             ({'compute_heuristic_representation': True}, False)])
def test_fourier_output_parameters_determine_whether_wavelet_can_follow(params, allowed):
    from types import SimpleNamespace
    from fedot_ind.core.optimizer.graph_validation import capabilities_from_runtime_node
    caps = industrial_capability_index()
    fourier = capabilities_from_runtime_node(SimpleNamespace(content={'name': 'fourier_basis', 'params': params}),
                                             capability_index=caps)
    model = OperationCapabilities('classifier', OperationKind.MODEL, ('tabular',), ('tabular',), ('classification',))
    graph = GraphSpec('basis', (GraphNodeSpec('f', 'fourier_basis', (), fourier),
                                GraphNodeSpec('w', 'wavelet_basis', ('f',), caps['wavelet_basis']),
                                GraphNodeSpec('q', 'quantile_extractor', ('w',), caps['quantile_extractor']),
                                GraphNodeSpec('m', 'classifier', ('q',), model)),
                      task_type='classification', input_data_type='ts')
    assert validate_graph(graph).is_valid is allowed
