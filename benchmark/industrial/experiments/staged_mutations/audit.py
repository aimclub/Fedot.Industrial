"""Feasibility of native catalog rewrites on TRAIN metadata, never fitness."""
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

from .config import load_protocol


def audit(output_dir):
    import torch
    from fedot.core.operations.operation import Operation
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot_ind.core.optimizer.graph_validation import ModelFeatureFlow
    from fedot_ind.core.optimizer.mutation_planning import (
        MutationKind, MutationLimits, AppliedMutation, apply_mutation_plan, enumerate_mutation_plans,
        candidate_signature)
    from fedot_ind.core.optimizer.mutation_runtime import (install_extension_aware_advisor, load_runtime_operation_catalog,
                                                           project_for_mutation, interpret_graph_spec, advisor_allows_plan)
    from fedot_ind.integration.fedot.evaluation import ConfirmedObjective
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope
    from benchmark.industrial.experiments.evolution_parallelism.search import build_environment
    from benchmark.industrial.experiments.mutation_diversity.data import load_training_data
    from benchmark.industrial.experiments.mutation_diversity.study import load_protocol as data_protocol

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    protocol = load_protocol()

    def torch_flags():
        return dict(threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                    deterministic=torch.are_deterministic_algorithms_enabled())
    flags_before = torch_flags()
    files = [*Path(__file__).parent.glob('*.py'), Path(__file__).with_name('protocol.json'),
             *Path('fedot_ind/core/optimizer').rglob('*.py'),
             Path('fedot_ind/core/optimizer/search_policy/defaults.json'),
             Path('fedot_ind/integration/fedot/evaluation.py')]
    sources = {str(path.resolve()): dict(sha256=sha256(path.read_bytes()).hexdigest(), source=path.read_text('utf-8'))
               for path in files}
    (root / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n', encoding='utf-8')
    rows = []
    with patch.object(Operation, 'fit', side_effect=AssertionError('Training forbidden')) as fit:
        with industrial_extension_scope():
            for task, scenario in protocol['tasks'].items():
                data = load_training_data(task, data_protocol())
                confirmed = ConfirmedObjective.from_training_data(lambda graph: None, tuple(x for x, y in data.folds))
                _, params = build_environment(protocol, data.refit.task, 1)
                builder = PipelineBuilder()
                for operation in scenario['initial_chain']:
                    builder.add_node(operation, params=protocol['defaults'].get(operation))
                graph = params.adapter.adapt(builder.build())
                install_extension_aware_advisor(params)
                params.verifier.input_data_type = confirmed.raw_input.data_type
                params.verifier.model_feature_flow = ModelFeatureFlow.TENSOR_FEATURES
                spec = project_for_mutation(graph, params)
                limits = MutationLimits(protocol['max_nodes'], protocol['max_depth'], protocol['max_candidates'],
                                        max_arity=protocol['max_arity'])
                catalog = load_runtime_operation_catalog(params)
                candidates = enumerate_mutation_plans(spec, catalog.capabilities, limits=limits,
                                                      kinds=(MutationKind.REWRITE,), raw_input=confirmed.raw_input)

                def topology_key(graph):
                    return candidate_signature(replace(graph, nodes=tuple(
                        replace(node, operation_name='node', capabilities=None) for node in graph.nodes)))

                initial_topology = topology_key(spec)
                topologies = {initial_topology}
                admissible, runtime_rejections = [], Counter()
                for plan in candidates.plans:
                    result = apply_mutation_plan(spec, plan, limits=limits, raw_input=confirmed.raw_input)
                    if isinstance(result, AppliedMutation) and advisor_allows_plan(spec, result.graph, graph, params.advisor):
                        runtime = interpret_graph_spec(graph, spec, result.graph)
                        if params.verifier(runtime):
                            topology = topology_key(result.graph)
                            topologies.add(topology)
                            admissible.append(dict(plan=plan.to_record(), graph=result.graph.to_record()))
                        else:
                            runtime_rejections.update(params.verifier.last_report.error_codes)
                    else:
                        runtime_rejections.update(('advisor_rejected',))
                rows.append(dict(task=task, dataset=scenario['name'], source=data.source,
                                 initial_graph=spec.to_record(), distinct_topologies=len(topologies),
                                 new_topologies=len(topologies - {initial_topology}),
                                 examined=candidates.examined, truncated=candidates.truncated,
                                 admissible_after_runtime=len(admissible),
                                 families=dict(Counter(item['plan']['edits'][0]['kind'] for item in admissible)),
                                 runtime_rejection_counts=dict(runtime_rejections),
                                 planning=candidates.diagnostics_record(), proposals=admissible))
        assert fit.call_count == 0
    if any(sha256(Path(path).read_bytes()).hexdigest() != record['sha256'] for path, record in sources.items()):
        raise RuntimeError('Audit sources changed')
    if torch_flags() != flags_before:
        raise RuntimeError('Global Torch settings changed')
    result = dict(protocol=protocol, training_calls=fit.call_count, rows=rows,
                  torch_flags_before=flags_before, torch_flags_after=torch_flags(),
                  note='Single-step feasibility only; no training, quality or speedup measurement.')
    (root / 'audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result
