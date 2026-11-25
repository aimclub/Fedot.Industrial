"""Bounded TSC execution; workers receive only already verified fold payloads."""

from dataclasses import asdict
import os
from pathlib import Path
from random import Random
from time import time

from ..cyclic_mutations.history import atomic_text
from ...protocols.json_io import parse_protocol_json
from .data import canonical_bytes, digest, read_development
from .runtime import Job, ValidationFold, evaluate
from .space import initial, proposals, rank
from .training import Recipe, recipe_grid, training_policy


def write_record(path, value):
    atomic_text(path, canonical_bytes(value).decode())


def read_record(path):
    return parse_protocol_json(Path(path).read_text(encoding='utf-8'))


def _batch(session, jobs, budget, seed, deadline, evaluator, attempts):
    from fedot_ind.core.optimizer.evaluation.contracts import EvaluationTask
    from fedot_ind.core.optimizer.evaluation.dask_config import DaskEvaluationPlan
    from fedot_ind.core.optimizer.evaluation.dask_executor import execute_dask_plan

    tasks = tuple(EvaluationTask(key, seed) for key, _ in jobs)
    plan = DaskEvaluationPlan(tasks, max_workers=session.max_workers, candidate_timeout_seconds=budget.candidate_seconds,
                              total_timeout_seconds=max(.001, deadline - time()))
    return execute_dask_plan(plan, (job for _, job in jobs), evaluator, session=session, on_attempt=attempts.append)


def _history_graph(graph):
    from golem.core.optimisers.graph import OptGraph, OptNode

    roots = []
    for branch in graph.branches:
        nodes = []
        for name in branch.operations:
            nodes = [OptNode({'name': name, 'params': {'parameter_index': branch.parameter}
                              if name == branch.head else {}}, nodes_from=nodes)]
        roots.extend(nodes)
    return OptGraph(roots[0] if len(roots) == 1 else OptNode('pilot_probability_mean', nodes_from=roots))


def _folds(panel, receipt):
    return tuple(ValidationFold(panel.take(fold['fit']), panel.take(fold['score']), tuple(fold['epoch_validation']))
                 for fold in receipt['inner'])


def _result_rows(graphs, results):
    return [dict(graph=asdict(graph), **asdict(result)) for graph, result in zip(graphs, results, strict=True)]


def cuda_admission(session, data, output, budget, seed, receipt_sha256):
    panel, description = read_development(data, receipt_sha256)
    folds = _folds(panel, description)[:1]
    policy = training_policy()
    recipe = Recipe(policy.learning_rates[0], policy.batches[1])
    graphs = tuple(graph for graph in initial() if graph.requires_cuda)
    jobs = tuple((graph.key, Job(graph, folds, tuple(description['classes']),
                                 {graph.branches[0].head: recipe}, 'cuda:0')) for graph in graphs)
    attempts = []
    results = _batch(session, jobs, budget, seed, time() + budget.search_seconds, evaluate, attempts)
    write_record(output / 'cuda_admission.json', dict(results=_result_rows(graphs, results),
                                                      attempts=[asdict(item) for item in attempts]))
    return all(result.status.value == 'succeeded' and result.value['folds'][0]['training']
               and all(row['device'] == 'cuda:0' for row in result.value['folds'][0]['training'])
               for result in results)


def calibrate(session, data, output, budget, seed, receipt_sha256, *, device='cuda:0', evaluator=evaluate):
    panel, description = read_development(data, receipt_sha256)
    folds = _folds(panel, description)
    training_sizes = [count for fold in folds for count in (len(fold.fit.rows),
                                                            len(fold.fit.rows) - len(fold.epoch_validation))]
    policy = training_policy()
    jobs, aliases, attempts = {}, [], []
    for graph in initial():
        if not graph.requires_cuda:
            continue
        family = graph.branches[0].head
        for nominal, actual in recipe_grid(policy, training_sizes):
            key = f'{family}-{actual.learning_rate}-{actual.batch_size}'
            aliases.append(dict(family=family, nominal=asdict(nominal), candidate=key))
            jobs.setdefault(key, Job(graph, folds, tuple(description['classes']), {family: actual}, device))
    deadline = float(os.environ.get('INDUSTRIAL_SEARCH_DEADLINE_UNIX', time() + budget.search_seconds))
    results = _batch(session, tuple(jobs.items()), budget, seed, deadline, evaluator, attempts)
    write_record(output / 'calibration.json', dict(development_sha256=receipt_sha256, aliases=aliases,
                                                   results=[asdict(result) for result in results],
                                                   attempts=[asdict(item) for item in attempts]))
    successful = {}
    for job, result in zip(jobs.values(), results, strict=True):
        if result.status.value == 'succeeded':
            (family, recipe), = job.recipes.items()
            successful.setdefault(family, []).append((result.value['loss'], recipe.learning_rate, recipe.batch_size))
    families = {family for job in jobs.values() for family in job.recipes}
    if successful.keys() != families:
        raise RuntimeError('At least one architecture has no successful recipe')
    decision = dict(development_sha256=receipt_sha256,
                    recipes={family: asdict(Recipe(*min(rows)[1:])) for family, rows in successful.items()})
    write_record(output / 'recipes.json', decision)
    return decision


def search(session, data, output, budget, seed, receipt_sha256, recipe_decision, *,
           device='cuda:0', evaluator=evaluate):
    from golem.core.optimisers.fitness.fitness import SingleObjFitness
    from golem.core.optimisers.objective.objective import ObjectiveInfo
    from golem.core.optimisers.opt_history_objects.individual import Individual
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory

    if recipe_decision['development_sha256'] != receipt_sha256:
        raise ValueError('Recipes are not bound to the admitted development data')
    recipes = {family: Recipe(**row) for family, row in recipe_decision['recipes'].items()}
    panel, description = read_development(data, receipt_sha256)
    folds = _folds(panel, description)
    history = OptHistory(ObjectiveInfo(metric_names=('balanced_accuracy_loss',)), default_save_dir=output)
    deadline = float(os.environ.get('INDUSTRIAL_SEARCH_DEADLINE_UNIX', time() + budget.search_seconds))
    rng, seen, attempts, evaluated, generation_records = Random(seed), set(), [], {}, []
    native = {}
    population = ()
    for generation in range(budget.generations + 1):
        if time() >= deadline:
            break
        tuning = generation in budget.tuning_generations
        kept = population[:budget.elites]
        sources = tuple(graph for graph, _ in rank(evaluated.items())) if tuning else tuple(g for g, _ in kept)
        expected = budget.population - len(kept)
        graphs = initial() if not generation else proposals(sources, seen, expected, tune=tuning, rng=rng)
        if len(graphs) != expected:
            generation_records.append(dict(index=generation, status='insufficient_unique_proposals'))
            break
        jobs = tuple((graph.key, Job(graph, folds, tuple(description['classes']), recipes, device))
                     for graph in graphs)
        results = _batch(session, jobs, budget, seed, deadline, evaluator, attempts)
        write_record(output / f'generation-{generation:02d}.json', _result_rows(graphs, results))
        current = tuple((graph, result.value['loss']) for graph, result in zip(graphs, results, strict=True)
                        if result.status.value == 'succeeded')
        for graph, loss in current:
            native[graph] = Individual(_history_graph(graph), fitness=SingleObjFitness(loss), uid=graph.key)
        seen.update(graph.key for graph in graphs)
        evaluated.update(current)
        population = rank((*kept, *current))
        row = dict(index=generation, phase='tuning' if tuning else 'structure' if generation else 'initial',
                   candidates=len(graphs), evaluated=len(current), carried=len(kept))
        generation_records.append(row)
        if population:
            history.add_to_history([native[graph] for graph, _ in population], generation_metadata=row)
            history.add_to_archive_history([native[graph] for graph, _ in rank(evaluated.items())[:budget.elites]])
            atomic_text(output / 'opt_history.json', history.save())
        if len(current) != expected:
            break
    complete = len(generation_records) == budget.generations + 1 and all(
        row.get('candidates') == row.get('evaluated') and 'candidates' in row for row in generation_records)
    winner = None
    if evaluated:
        graph, loss = rank(evaluated.items())[0]
        winner = dict(graph=asdict(graph), graph_sha256=graph.key, loss=loss)
    result = dict(status='complete' if complete else 'incomplete', dataset=panel.dataset, seed=seed,
                  official_test_used=False, development_sha256=receipt_sha256, generations=generation_records,
                  attempts=[asdict(item) for item in attempts], winner=winner,
                  recipes_sha256=digest(canonical_bytes(recipe_decision)))
    write_record(output / 'search.json', result)
    return result
