"""Coordinator-owned bounded family archive and stage application."""
from copy import deepcopy

from golem.core.optimisers.opt_history_objects.individual import Individual

from fedot_ind.core.optimizer.mutation import mutation_name
from .parameters import BoundedParameterMutation, rewrite_subgraph_mutation
from .planning import FamilyCandidate, action_weights, select_survivors, stage_for_budget


def configure_actions(parameters, policy):
    """Clone caller parameters; unrestricted parameter callbacks are not retained."""
    result = deepcopy(parameters)
    actions = [action for action in result.mutation_types
               if mutation_name(action) not in ('parameter_change_mutation', 'rewrite_subgraph_mutation')]
    actions.append(BoundedParameterMutation(policy.parameters))
    if any(stage.weights.rewrite > 0 for stage in policy.stages):
        actions.append(rewrite_subgraph_mutation)
    action_weights(actions, policy.stages[0])
    result.mutation_types = actions
    return result


class StagedSearchController:
    def __init__(self, policy, graph_generation_params, budget_seconds):
        stage_for_budget(policy, 0., budget_seconds)
        self.policy = policy
        self.params = graph_generation_params
        self.budget_seconds = budget_seconds
        self.stage = policy.stages[0]
        self.events = []
        self.families = {}
        self._last_elapsed = 0.

    def apply(self, mutation, elapsed_seconds):
        from golem.core.optimisers.adaptive.operator_agent import RandomAgent
        stage = stage_for_budget(self.policy, elapsed_seconds, self.budget_seconds)
        if elapsed_seconds < self._last_elapsed:
            raise ValueError('Elapsed search time must not move backwards')
        if not isinstance(mutation.agent, RandomAgent):
            raise ValueError('Staged search requires a RandomAgent')
        mutation.agent._probs = action_weights(mutation.agent.available_actions, stage)
        self._last_elapsed = elapsed_seconds
        self.stage = stage
        if not self.events or self.events[-1]['name'] != stage.name:
            self.events.append(dict(name=stage.name, elapsed_seconds=elapsed_seconds,
                                    budget_seconds=self.budget_seconds, weights=stage.weights.to_record(),
                                    family_fraction=stage.family_fraction))

    def _description(self, individual):
        from fedot_ind.core.optimizer.mutation_runtime import project_for_mutation
        from fedot_ind.core.optimizer.mutation_planning._graph import candidate_signature
        spec = project_for_mutation(individual.graph, self.params)
        context = individual.metadata.get('evaluation_context') or {}
        key = context.get('configuration_id') or individual.graph.descriptive_id
        return FamilyCandidate(key, candidate_signature(spec), float(individual.fitness.value), len(spec.nodes))

    @staticmethod
    def _copy(individual):
        return Individual(graph=deepcopy(individual.graph), uid=individual.uid,
                          fitness=deepcopy(individual.fitness), metadata=deepcopy(individual.metadata))

    def remember(self, individual):
        description = self._description(individual)
        previous = self.families.get(description.family)
        def rank(candidate): return (candidate.loss, candidate.nodes, candidate.key)
        if previous is None or rank(description) < rank(previous[0]):
            self.families[description.family] = (description, self._copy(individual))
        if len(self.families) > self.policy.max_families:
            worst = max(self.families, key=lambda family: rank(self.families[family][0]))
            del self.families[worst]

    def survivors(self, population, capacity):
        descriptions, individuals = {}, {}
        for individual in [*(item[1] for item in self.families.values()), *population]:
            if not individual.fitness.valid:
                continue
            description = self._description(individual)
            old = descriptions.get(description.key)
            if old is None or description.loss <= old.loss:
                descriptions[description.key], individuals[description.key] = description, individual
        keys = select_survivors(tuple(descriptions.values()), capacity, self.stage.family_fraction)
        return [individuals[key] for key in keys]

    def export(self, output_dir):
        import json
        from pathlib import Path
        record = dict(schema_version='industrial_staged_search_trace@1', policy=self.policy.to_record(),
                      events=self.events, families=[dict(family=d.family, configuration=d.key,
                                                         loss=d.loss, nodes=d.nodes) for d, _ in self.families.values()])
        target = Path(output_dir) / 'mutation_schedule.json'
        target.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        return target
