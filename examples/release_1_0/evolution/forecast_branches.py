"""Generate and verify two to four forecast branches; never fit a model.

    python -m examples.release_1_0.evolution.forecast_branches --output-dir <new-directory>
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch


def build_branch_graphs():
    from fedot.core.pipelines.node import PipelineNode
    from fedot_ind.core.optimizer.graph_validation import (
        GraphNodeSpec, GraphSpec, ModelFeatureFlow, capabilities_from_runtime_node, industrial_capability_index)
    from fedot_ind.core.optimizer.mutation_planning import (
        AppliedMutation, ForkForecastBranch, MutationLimits, MutationPlan, RawInputSpec,
        apply_mutation_plan, graph_signature)
    settings = json.loads(Path(__file__).with_suffix('.json').read_text('utf-8'))
    if settings['training_enabled'] is not False:
        raise ValueError('This example does not permit training')
    caps = industrial_capability_index()
    models = tuple(capabilities_from_runtime_node(PipelineNode(name), capability_index=caps)
                   for name in settings['models'])
    if not models or any(model is None or 'ts_forecasting' not in model.task_types
                         or 'tabular' not in model.input_data_types for model in models):
        raise ValueError('Each declared branch model must support tabular forecasting windows')
    ridge = models[0]
    initial = GraphSpec('forecast-branches', (
        GraphNodeSpec('window', 'hankelisation', (), caps['hankelisation']),
        GraphNodeSpec('model', ridge.name, ('window',), ridge)), task_type='ts_forecasting', input_data_type='ts',
        model_feature_flow=ModelFeatureFlow.TENSOR_FEATURES)
    raw = RawInputSpec('ts', True)
    limits = MutationLimits(settings['max_nodes'], settings['max_depth'], max_arity=settings['max_arity'])
    graphs = []
    for merge_name in settings['merges']:
        graph = initial
        for count, representation in enumerate(settings['representations'], 2):
            merge_id = next((n.node_id for n in graph.nodes if n.operation_name == merge_name), 'model')
            model = models[(count - 2) % len(models)]
            operations = (caps[representation],) + ((model,) if merge_name == 'forecast_mean' else ())
            edit = ForkForecastBranch(merge_id, 'window', operations, caps[merge_name])
            result = apply_mutation_plan(graph, MutationPlan(graph_signature(graph), (edit,), raw),
                                         limits=limits, raw_input=raw)
            if not isinstance(result, AppliedMutation):
                raise ValueError(f'Invalid branch graph: {result}')
            graphs.append((merge_name, count, result.graph))
            graph = result.graph
    return settings, tuple(graphs)


def restore_branch_graph(graph, settings):
    from golem.core.optimisers.graph import OptGraph, OptNode
    from fedot_ind.integration.fedot.forecast_branches import ForecastPipelineAdapter
    nodes = {}
    for node in graph.nodes:
        params = {}
        if node.operation_name == 'hankelisation':
            params = {key: settings[key] for key in ('window_size', 'stride', 'forecast_length')}
        elif node.operation_name == 'forecast_mean':
            params = {'forecast_length': settings['forecast_length']}
        nodes[node.node_id] = OptNode({'name': node.operation_name, 'params': params})
    for node in graph.nodes:
        nodes[node.node_id].nodes_from = [nodes[key] for key in node.parent_ids]
    return ForecastPipelineAdapter().restore(OptGraph(list(nodes.values())))


def audit_mutation_routes(settings):
    """Check actual positional projection and advisor vetoes inside an extension scope."""
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot.core.pipelines.verification import verifier_for_task
    from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
    from benchmark.industrial.experiments.evolution_parallelism.search import build_environment
    from fedot_ind.core.optimizer.graph_validation import IndustrialGraphVerifier, ModelFeatureFlow
    from fedot_ind.core.optimizer.mutation_planning import (
        AppliedMutation, ForkForecastBranch, MutationLimits, MutationPlan, RawInputSpec,
        apply_mutation_plan, graph_signature)
    from fedot_ind.core.optimizer.mutation_runtime import (
        advisor_allows_plan, install_extension_aware_advisor, interpret_graph_spec,
        load_runtime_operation_catalog, project_for_mutation)
    from fedot_ind.integration.fedot.forecast_branches import ForecastPipelineAdapter

    environment = {'tasks': {'ts_forecasting': {'operations': [
        'hankelisation', *settings['models'], *settings['merges'], *settings['representations']]}},
        'max_depth': settings['max_depth'], 'max_arity': settings['max_arity'], 'budget_seconds': 30}
    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(settings['forecast_length']))
    _, params = build_environment(environment, task, 1)
    params.adapter = ForecastPipelineAdapter()
    params.verifier = IndustrialGraphVerifier(
        adapter=params.adapter, task_type='ts_forecasting', input_data_type='ts',
        model_feature_flow=ModelFeatureFlow.TENSOR_FEATURES,
        legacy_verifier=verifier_for_task(TaskTypesEnum.ts_forecasting, params.adapter))
    install_extension_aware_advisor(params)
    catalog = {op.name: op for op in load_runtime_operation_catalog(params).capabilities}
    raw = RawInputSpec('ts', True)
    limits = MutationLimits(settings['max_nodes'], settings['max_depth'], max_arity=settings['max_arity'])
    records = []
    for merge in settings['merges']:
        initial = PipelineBuilder().add_node('hankelisation', params={key: settings[key] for key in (
            'window_size', 'stride', 'forecast_length')}).add_node(settings['models'][0]).build()
        graph = params.adapter.adapt(initial)
        for count, name in enumerate(settings['representations'], 2):
            before = project_for_mutation(graph, params)
            child = next((n for n in before.nodes if n.operation_name == merge),
                         next(n for n in before.nodes if n.operation_name == settings['models'][0]))
            window = next(n for n in before.nodes if n.operation_name == 'hankelisation')
            model = catalog[settings['models'][(count - 2) % len(settings['models'])]]
            operations = (catalog[name],) + ((model,) if merge == 'forecast_mean' else ())
            edit = ForkForecastBranch(child.node_id, window.node_id, operations, catalog[merge])
            plan = MutationPlan(graph_signature(before), (edit,), raw)
            outcome = apply_mutation_plan(before, plan, limits=limits, raw_input=raw)
            if not isinstance(outcome, AppliedMutation):
                raise ValueError(f'Invalid runtime proposal: {outcome}')
            advised = advisor_allows_plan(before, outcome.graph, graph, params.advisor)
            graph = interpret_graph_spec(graph, before, outcome.graph)
            valid = bool(params.verifier(graph))
            records.append({'merge': merge, 'branch_count': count, 'advisor': advised, 'native_verifier': valid,
                            'plan': plan.to_record(), 'source': before.to_record(), 'graph': outcome.graph.to_record()})
            if not advised or not valid:
                raise ValueError(
                    f'Runtime refused {merge}/{count}: advisor={advised}, report={params.verifier.last_report}')
    return records


def audit_branches():
    import torch
    from fedot.core.operations.operation import Operation
    from fedot.core.pipelines.verification import verifier_for_task
    from fedot.core.repository.tasks import TaskTypesEnum
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope
    from fedot_ind.integration.fedot.forecast_branches import ForecastPipelineAdapter
    before = (torch.get_num_threads(), torch.get_num_interop_threads(), torch.are_deterministic_algorithms_enabled())
    records = []
    with patch.object(Operation, 'fit', side_effect=AssertionError('Training forbidden')) as fit:
        with industrial_extension_scope():
            settings, graphs = build_branch_graphs()
            verifier = verifier_for_task(TaskTypesEnum.ts_forecasting, ForecastPipelineAdapter())
            for merge, count, graph in graphs:
                pipeline = restore_branch_graph(graph, settings)
                valid = bool(verifier(pipeline))
                records.append({'merge': merge, 'branch_count': count, 'native_verifier': valid,
                                'graph': graph.to_record()})
            mutation_routes = audit_mutation_routes(settings)
        assert fit.call_count == 0
    after = (torch.get_num_threads(), torch.get_num_interop_threads(), torch.are_deterministic_algorithms_enabled())
    if before != after:
        raise RuntimeError('Torch settings changed')
    source_paths = [Path(__file__), Path(__file__).with_suffix('.json'),
                    *Path('fedot_ind/core/models/ts_forecasting/branching').glob('*.py'),
                    Path('fedot_ind/integration/fedot/forecast_branches.py'),
                    Path('fedot_ind/integration/fedot/forecast_merge_context.py'),
                    Path('fedot_ind/integration/fedot/extensions/factories.py'),
                    Path('fedot_ind/integration/fedot/extensions/discovery.py'),
                    Path('fedot_ind/integration/fedot/extensions/catalog.json'),
                    Path('fedot_ind/core/repository/data/default_operation_params.json'),
                    *Path('fedot_ind/core/optimizer/mutation_planning').glob('*.py'),
                    *Path('fedot_ind/core/optimizer/graph_validation').glob('*.py'),
                    Path('fedot_ind/core/optimizer/mutation_runtime.py')]
    return {'schema_version': 'industrial_forecast_branch_audit@1', 'training_calls': fit.call_count,
            'official_test_used': False, 'dataset_training_performed': False,
            'torch_before': before, 'torch_after': after, 'settings': settings, 'graphs': records,
            'mutation_routes': mutation_routes,
            'sources': {str(p): {'sha256': sha256(p.read_bytes()).hexdigest(), 'source': p.read_text('utf-8')}
                        for p in source_paths},
            'note': 'Structural feasibility only. No quality, latency or model training measurement.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    report = audit_branches()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        with (args.output_dir / 'audit.json').open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
    for item in report['graphs']:
        print(f"{item['merge']}: branches={item['branch_count']}, native_verifier={item['native_verifier']}")
    print('Operation.fit calls:', report['training_calls'])
    if not all(item['native_verifier'] for item in report['graphs']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
