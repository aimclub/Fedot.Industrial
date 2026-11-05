"""Cyclic phase application and bounded, multi-family refinement roots."""
from math import ceil

from .runtime import StagedSearchController


class CyclicSearchController(StagedSearchController):
    def __init__(self, policy, graph_generation_params, budget_seconds):
        super().__init__(policy, graph_generation_params, budget_seconds)
        self.locked_families = ()
        self.cursor = 0
        self._forced_index = 0

    def apply(self, mutation, elapsed_seconds):
        from .planning import stage_for_budget
        if elapsed_seconds < self._last_elapsed:
            raise ValueError('Elapsed search time must not move backwards')
        natural = stage_for_budget(self.policy, elapsed_seconds, self.budget_seconds)
        # Early frontier exhaustion can advance, never rewind, the global schedule.
        effective = max(self.policy.stages.index(natural), self._forced_index)
        effective_elapsed = max(elapsed_seconds, 0. if effective == 0 else
                                self.policy.stages[effective - 1].until * self.budget_seconds)
        previous = self.stage.name if self.events else None
        super().apply(mutation, effective_elapsed)
        self._last_elapsed = elapsed_seconds
        from fedot_ind.core.optimizer.randomness import SeededOperator
        planned = mutation
        while isinstance(planned, SeededOperator):
            planned = planned.operator
        planned.structure_locked = self.stage.conservative
        if previous != self.stage.name:
            self.events[-1]['elapsed_seconds'] = elapsed_seconds
            index = self.policy.stages.index(self.stage)
            expected_start = 0. if index == 0 else self.policy.stages[index - 1].until * self.budget_seconds
            self.events[-1].update(expected_start_seconds=expected_start,
                                   switch_lag_seconds=max(0., elapsed_seconds - expected_start))
            if self.stage.conservative:
                ranked = sorted(self.families, key=lambda f: (
                    self.families[f][0].loss, self.families[f][0].nodes, self.families[f][0].key))
                capacity = max(2, ceil(planned.operator.parameters.pop_size * self.stage.family_fraction))
                self.locked_families = tuple(ranked[:capacity])
                self.cursor = 0
                self.events[-1]['locked_families'] = list(self.locked_families)

    def frontier_exhausted(self):
        index = self.policy.stages.index(self.stage)
        self.events[-1]['frontier_exhausted'] = True
        if index == len(self.policy.stages) - 1:
            return False
        self._forced_index = index + 1
        return True

    def refinement_parents(self, count):
        if not self.stage.conservative:
            raise ValueError('Refinement roots are only available in a conservative phase')
        roots = [self.families[key][1] for key in self.locked_families if key in self.families]
        if not roots:
            return []
        result = [roots[(self.cursor + i) % len(roots)] for i in range(count)]
        self.cursor += count
        return result
