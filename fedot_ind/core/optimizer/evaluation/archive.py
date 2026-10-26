"""Coordinator-owned evidence and final choices from completed evaluations."""

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite

from golem.core.optimisers.archive import GenerationKeeper
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.generation import Generation

from fedot_ind.core.optimizer.domain import EvaluationSucceeded, evaluation_outcome_from_metadata
from fedot_ind.core.optimizer.observability import graph_identity
from .preparation import PreparedEvaluationGraph, canonical_json


@dataclass(frozen=True)
class CompletedEvaluation:
    evaluation_id: str
    candidate_id: str
    fitness: tuple[float, ...]
    graph_id: str
    context_json: str | None

    def to_record(self):
        import json
        return {'evaluation_id': self.evaluation_id, 'candidate_id': self.candidate_id,
                'fitness': list(self.fitness), 'graph_id': self.graph_id,
                'evaluation_context': None if self.context_json is None else json.loads(self.context_json)}


def completed_evaluation(result, sequence):
    """Only a finite, successful, freshly completed result can enter the archive."""
    if result is None or result.metadata.get('evaluation_status') != 'succeeded':
        return None
    if not isinstance(evaluation_outcome_from_metadata(result.metadata), EvaluationSucceeded):
        return None
    fitness = result.fitness
    if not fitness.valid or not fitness.values or not all(isfinite(value) for value in fitness.values):
        return None
    candidate_id = str(result.uid_of_individual)
    context = result.metadata.get('evaluation_context')
    if context is None:
        evaluation_id, context_json = f'{candidate_id}:archive:{sequence}', None
    else:
        if (type(context) is not dict or context.get('candidate_id') != candidate_id
                or type(context.get('evaluation_id')) is not str or not context['evaluation_id']):
            raise ValueError('Completed evaluation context belongs to another candidate')
        configuration = context.get('configuration')
        if configuration is not None:
            prepared = PreparedEvaluationGraph(result.graph, canonical_json(configuration))
            if prepared.configuration_id != context.get('configuration_id'):
                raise ValueError('Completed evaluation configuration identity differs')
        evaluation_id, context_json = context['evaluation_id'], canonical_json(context)
    return CompletedEvaluation(evaluation_id, candidate_id, tuple(map(float, fitness.values)),
                               graph_identity(result.graph), context_json)


def final_history_choices(generations, choices):
    """Reuse historical UID owners so final snapshots cannot erase their lineage."""
    known, pending = {}, [item for generation in generations for item in generation]
    while pending:
        individual = pending.pop()
        if individual.uid in known:
            continue
        known[individual.uid] = individual
        pending.extend(individual.parents)
    result = []
    for choice in choices:
        original = known.get(choice.uid)
        if original is not None:
            if (tuple(original.fitness.values) != tuple(choice.fitness.values)
                    or graph_identity(original.graph) != graph_identity(choice.graph)
                    or original.metadata.get('evaluation_context') != choice.metadata.get('evaluation_context')):
                raise ValueError('Final snapshot conflicts with the historical UID owner')
            result.append(original)
        else:
            result.append(choice)
    return tuple(result)


def record_final_history(history, choices, metadata=None):
    """Publish a result group without assigning a fabricated native generation."""
    historical = final_history_choices(history.generations, choices)
    group = Generation((), history.generations_count, 'final_choices', metadata)
    # Generation's normal append/set path assigns native_generation to parents.
    group.data.extend(historical)
    history.generations.append(group)
    history.add_to_archive_history(historical)


class CompletedEvaluationArchive:
    """Retain detached lightweight graphs, never parent lineage or fitted pipelines.

    The adapter/cleanup boundary supplies optimisation graphs. Final single-objective
    selection uses the exact lexicographic minimum, keeping first-seen ties. Multiple
    objectives retain GOLEM's ParetoFront. Neither advances generation counters.
    Storage is O(completed evaluations * graph size); accessors return copies.
    """

    def __init__(self, objective, keep_n_best=1):
        self._entries = {}
        self._candidates = {}
        self._multi_objective = objective.is_multi_objective
        self._keep_n_best = keep_n_best or 0
        self._best = GenerationKeeper(objective, keep_n_best=keep_n_best).archive

    @property
    def entries(self):
        return tuple(self._entries.values())

    @staticmethod
    def _snapshot(candidate):
        return Individual(graph=deepcopy(candidate.graph), uid=candidate.uid,
                          fitness=deepcopy(candidate.fitness), metadata=deepcopy(candidate.metadata))

    @property
    def best_individuals(self):
        candidates = self._best.items if self._multi_objective else sorted(
            self._candidates.values(), key=lambda item: tuple(item.fitness.values))[:self._keep_n_best]
        return tuple(self._snapshot(candidate) for candidate in candidates)

    def record_results(self, results):
        for result in results:
            entry = completed_evaluation(result, len(self._entries))
            if entry is None:
                continue
            existing = self._entries.get(entry.evaluation_id)
            if existing is not None:
                if existing != entry:
                    raise ValueError('Completed evaluation episode has conflicting evidence')
                continue
            candidate = Individual(graph=deepcopy(result.graph), uid=entry.candidate_id,
                                   fitness=deepcopy(result.fitness), metadata=deepcopy(result.metadata))
            candidate.metadata['archive_evaluation_id'] = entry.evaluation_id
            self._entries[entry.evaluation_id] = entry
            self._candidates[entry.evaluation_id] = candidate
            if self._multi_objective:
                self._best.update([candidate])
