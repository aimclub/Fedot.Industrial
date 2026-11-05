"""Opt-in bounded reproduction, with phase decisions between evaluation batches."""
from dataclasses import replace
import json
from time import perf_counter

from golem.core.optimisers.genetic.operators.reproduction import ReproductionController
from golem.core.optimisers.graph import OptGraph
from golem.utilities.data_structures import ensure_wrapped_in_sequence

from .parameters import BoundedParameterMutation


class ParameterFrontier:
    """Novelty only; no fitness cache or reuse of fitted models."""

    def __init__(self, preparation):
        self.preparation = getattr(preparation, 'base', preparation)
        self.submitted = set()

    def identity(self, graph):
        if not isinstance(graph, OptGraph):
            graph = self.preparation.adapter.adapt(graph)
        return self.preparation(graph).configuration_id

    def __call__(self, graph):
        return self.identity(graph) not in self.submitted


class BoundedBatchReproduction(ReproductionController):
    def __init__(self, original, owner, batch_size):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('Reproduction batch size must be positive')
        super().__init__(original.parameters, original.selection, original.mutation, original.crossover,
                         window_size=original._window_size)
        self._success_rate_window = original._success_rate_window.copy()
        self.owner, self.batch_size = owner, batch_size
        self.frontier = ParameterFrontier(owner.eval_dispatcher.graph_preparation)
        self.batches = []
        agent = owner.mutation.agent
        agent.actions = [replace(action, eligible=self.frontier) if isinstance(action, BoundedParameterMutation)
                         else action for action in agent.available_actions]
        from fedot_ind.core.optimizer.randomness import SeededOperator
        planned = owner.mutation
        while isinstance(planned, SeededOperator):
            planned = planned.operator
        previous = planned.candidate_observer

        def reserve(candidate):
            self.frontier.submitted.add(self.frontier.identity(candidate.graph))
            if previous is not None:
                previous(candidate)
        planned.candidate_observer = reserve

    def reproduce_uncontrolled(self, population, evaluator, pop_size=None):
        from .cyclic_runtime import CyclicSearchController
        controller = self.owner.staged_search
        self.frontier.submitted.update(json.loads(entry.context_json)['configuration_id']
                                       for entry in self.owner.completed_evaluations.entries if entry.context_json)
        target = len(population) if pop_size is None else pop_size
        evaluated = []
        for start in range(0, target, self.batch_size):
            if self.owner.timer.is_time_limit_reached():
                break
            controller.apply(self.owner.mutation, self.owner.timer.minutes_from_start * 60.)
            count = min(self.batch_size, target - start)
            conservative = isinstance(controller, CyclicSearchController) and controller.stage.conservative
            parents = (controller.refinement_parents(count) if conservative else self.selection(population, count))
            children = ensure_wrapped_in_sequence(self.mutation(self.crossover(parents))) if parents else []
            if conservative and not children:
                self.batches.append(dict(phase=controller.stage.name, candidates=0, frontier_exhausted=True))
                if not controller.frontier_exhausted():
                    from fedot_ind.core.optimizer.evaluation.budget import SearchBudgetExhausted
                    raise SearchBudgetExhausted('Final conservative frontier exhausted; no further novel neighbours')
                continue
            for child in children:
                self.frontier.submitted.add(self.frontier.identity(child.graph))
            row = dict(phase=controller.stage.name, candidates=len(children), conservative=conservative,
                       elapsed_seconds=self.owner.timer.minutes_from_start * 60.)
            self.batches.append(row)
            started = perf_counter()
            results = evaluator(children) if children else []
            row.update(evaluation_seconds=perf_counter() - started, successful=len(results),
                       end_seconds=self.owner.timer.minutes_from_start * 60.)
            evaluated.extend(results)
        return evaluated


def install_bounded_reproduction(owner, batch_size):
    owner.reproducer = BoundedBatchReproduction(owner.reproducer, owner, batch_size)
