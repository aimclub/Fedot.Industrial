"""Real mutation proposals on fixed parents, with training forbidden."""
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
from copy import deepcopy

from .study import load_protocol, MutationStudy


def initial_configuration_ids(task):
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from benchmark.industrial.experiments.evolution_parallelism.config import load_defaults
    from benchmark.industrial.experiments.mutation_planning.population_study.preparation import (
        configure_pipeline, pipeline_configuration)
    from fedot_ind.core.optimizer.evaluation.preparation import canonical_json
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope
    settings = load_defaults()
    if task == 'ts_forecasting':
        settings['operation_parameters'].setdefault('hankelisation', {})['forecast_length'] = 12
    result = set()
    with industrial_extension_scope():
        for chain in settings['tasks'][task]['initial_chains']:
            builder = PipelineBuilder()
            for operation in chain:
                builder.add_node(operation, params=settings['operation_parameters'].get(operation))
            pipeline = configure_pipeline(builder.build(), settings['operation_parameters'])
            result.add(sha256(canonical_json(pipeline_configuration(pipeline, task)).encode()).hexdigest())
    return result


def run_generation(output_dir, *, attempts=None):
    from fedot.core.operations.operation import Operation
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
    from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
    from golem.core.optimisers.objective import Objective
    from golem.core.optimisers.opt_history_objects.individual import Individual
    from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
    from fedot_ind.core.optimizer.configuration import EvolutionConfig, MutationPlanningPolicy, ResourceBudget
    from fedot_ind.core.optimizer.evaluation.configuration import EvaluationExecutionConfig, EvaluationBackend
    from fedot_ind.core.optimizer.randomness import SeededOperator, operator_random_scope
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope
    from benchmark.industrial.experiments.evolution_parallelism.config import load_defaults
    from benchmark.industrial.experiments.evolution_parallelism.search import build_environment
    from benchmark.industrial.experiments.mutation_planning.population_study.preparation import (
        StudyGraphPreparation, pipeline_configuration)
    from fedot_ind.core.optimizer.evaluation.preparation import canonical_json

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    protocol, settings = load_protocol(), load_defaults()
    count = protocol['proposal_attempts'] if attempts is None else attempts
    rows = []
    sources = [Path(__file__), Path(__file__).with_name('protocol.json'),
               *Path('fedot_ind/core/optimizer').rglob('*.py'),
               Path('fedot_ind/core/optimizer/mutation_profiles.json'),
               Path('fedot_ind/core/repository/constanst_repository.py'),
               Path('benchmark/industrial/experiments/evolution_parallelism/defaults.json'),
               Path(__file__).with_name('data.py'), Path(__file__).with_name('study.py')]
    hashes = {str(p.resolve()): sha256(p.read_bytes()).hexdigest() for p in sources}
    (root / 'source_snapshot.json').write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
    (root / 'frozen_sources.json').write_text(json.dumps({str(p.resolve()): p.read_text('utf-8') for p in sources},
                                                         indent=2) + '\n', encoding='utf-8')
    with ExitStack() as stack:
        stack.enter_context(industrial_extension_scope())
        fit = stack.enter_context(patch.object(Operation, 'fit', side_effect=AssertionError('Training forbidden')))
        for task in protocol['generation_tasks']:
            data = MutationStudy('baseline', task).load_data()
            parameters = deepcopy(settings['operation_parameters'])
            if task == 'ts_forecasting':
                parameters.setdefault('hankelisation', {})['forecast_length'] = 12
            for arm in protocol['arms']:
                study = MutationStudy(arm, task)
                requirements, params = build_environment(settings, data.refit.task, 1)
                chains = settings['tasks'][task]['initial_chains']
                initial = []
                for chain in chains:
                    builder = PipelineBuilder()
                    for operation in chain:
                        builder.add_node(operation, params=parameters.get(operation))
                    initial.append(params.adapter.adapt(builder.build()))
                optimizer = IndustrialEvoOptimizer(Objective({'loss': lambda graph: 0.}), initial,
                                                   requirements, params, GPAlgorithmParameters(pop_size=8, crossover_prob=0., mutation_prob=1.,
                                                                                               variable_mutation_num=False,
                                                                                               mutation_types=[MutationTypesEnum.single_change, MutationTypesEnum.single_add, MutationTypesEnum.single_drop]),
                                                   EvolutionConfig(random_seed=42, mutation_strategy=study.strategy,
                                                                   mutation_planning=MutationPlanningPolicy(
                                                                       enabled=True, max_candidates=64, max_nodes=6),
                                                                   resource_budget=ResourceBudget(8, 12, 32),
                                                                   evaluation_execution=EvaluationExecutionConfig(EvaluationBackend.SEQUENTIAL)))
                mutation = optimizer.mutation
                while isinstance(mutation, SeededOperator):
                    mutation = mutation.operator
                prep = StudyGraphPreparation.from_parameters(params.adapter, parameters, task)
                for parent_index, graph in enumerate(initial):
                    parent = Individual(graph)
                    statuses, actions, reasons = Counter(), Counter(), Counter()
                    raw_ids, configs, structures, topologies = set(), set(), set(), set()
                    started = perf_counter()
                    with operator_random_scope(42):
                        for _ in range(count):
                            child = mutation._mutate(parent)
                            event = mutation.last_execution
                            statuses[event.status.value] += 1
                            actions[event.action or 'no_action'] += 1
                            if event.planning:
                                for item in event.planning.diagnostics:
                                    reasons.update(item.reason_codes)
                            if child is None or child is parent:
                                continue
                            before = pipeline_configuration(params.adapter.restore(child.graph), task)
                            raw_ids.add(canonical_json(before))
                            configuration = json.loads(prep(child.graph).configuration_json)
                            configs.add(canonical_json(configuration))
                            nodes = configuration['nodes']
                            structures.add(canonical_json(
                                [{'operation': n['operation'], 'parents': n['parents']} for n in nodes]))
                            topologies.add(canonical_json([n['parents'] for n in nodes]))
                    row = dict(task=task, arm=arm, parent=parent_index, attempts=count,
                               seconds=perf_counter() - started, statuses=dict(statuses), actions=dict(actions),
                               reasons=dict(reasons), raw_unique=len(raw_ids), configurations=len(configs),
                               structures=len(structures), topologies=len(topologies), raw_input_confirmed=False,
                               probabilities=optimizer.mutation_probability_report,
                               configuration_ids=sorted(sha256(c.encode()).hexdigest() for c in configs))
                    rows.append(row)
                    with (root / 'generation.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(row) + '\n')
                    print(json.dumps({k: row[k] for k in ('task', 'arm', 'parent', 'attempts',
                          'seconds', 'configurations', 'topologies')}), flush=True)
        assert fit.call_count == 0
    result = dict(protocol=protocol, settings=settings, rows=rows, training_calls=0,
                  initial_configuration_ids={task: sorted(initial_configuration_ids(task)) for task in protocol['generation_tasks']})
    if any(sha256(Path(p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
        raise RuntimeError('Generation implementation changed during the audit')
    (root / 'generation.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result
