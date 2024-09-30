from copy import deepcopy
from random import choice
from typing import Sequence

from golem.core.constants import MAX_GRAPH_GEN_ATTEMPTS
from golem.core.optimisers.adaptive.operator_agent import RandomAgent
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.operator import EvaluationOperator, PopulationT
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.populational_optimizer import _try_unfit_graph

from fedot_ind.core.repository.IndustrialDispatcher import IndustrialDispatcher
from fedot_ind.core.repository.constanst_repository import FEDOT_MUTATION_STRATEGY


class IndustrialEvoOptimizer(EvoGraphOptimizer):
    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[OptGraph],
                 requirements: GraphRequirements,
                 graph_generation_params: GraphGenerationParams,
                 graph_optimizer_params: GPAlgorithmParameters):

        for mutation in graph_optimizer_params.mutation_types:
            try:
                is_invalid = mutation.__name__.__contains__('resample')
            except Exception:
                is_invalid = mutation.name.__contains__('resample')
            if is_invalid:
                graph_optimizer_params.mutation_types.remove(mutation)

        graph_optimizer_params.adaptive_mutation_type = RandomAgent(actions=graph_optimizer_params.mutation_types,
                                                                    probs=FEDOT_MUTATION_STRATEGY[
                                                                        'params_mutation_strategy'])
        super().__init__(objective, initial_graphs, requirements,
                         graph_generation_params, graph_optimizer_params)
        self.operators.remove(self.crossover)
        self.requirements = requirements
        self.eval_dispatcher = IndustrialDispatcher(
            adapter=graph_generation_params.adapter,
            n_jobs=requirements.n_jobs,
            graph_cleanup_fn=_try_unfit_graph,
            delegate_evaluator=graph_generation_params.remote_evaluator)

    def _create_initial_population(self, initial_assumption):
        initial_individuals = [Individual(graph, metadata=self.requirements.static_individual_metadata)
                               for graph in initial_assumption]
        return initial_individuals

    def _initial_population(self, evaluator: EvaluationOperator):
        """ Initializes the initial population """
        # Adding of initial assumptions to history as zero generation
        pop_size = self.graph_optimizer_params.pop_size
        pop_label = 'initial_assumptions'
        self.initial_individuals = self._create_initial_population(self.initial_graphs)

        if len(self.initial_individuals) < pop_size:
            self.initial_individuals = self._extend_population(self.initial_individuals, pop_size)
            pop_label = 'extended_initial_assumptions'
        init_pop = evaluator(self.initial_individuals)
        self._update_population(init_pop, pop_label)

    def _extend_population(self, pop: PopulationT, target_pop_size: int) -> PopulationT:
        verifier = self.graph_generation_params.verifier
        extended_pop = list(pop)
        pop_graphs = [ind.graph for ind in extended_pop]
        # Set mutation probabilities to 1.0
        initial_req = deepcopy(self.requirements)
        initial_req.mutation_prob = 1.0
        self.mutation.update_requirements(requirements=initial_req)

        for iter_num in range(MAX_GRAPH_GEN_ATTEMPTS):
            if len(extended_pop) == target_pop_size:
                break
            new_ind = self.mutation(choice(pop))
            if new_ind:
                new_graph = new_ind.graph
                if new_graph not in pop_graphs and verifier(new_graph):
                    extended_pop.append(new_ind)
                    pop_graphs.append(new_graph)
        else:
            self.log.warning(f'Exceeded max number of attempts for extending initial graphs, stopping.'
                             f'Current size {len(pop)}, required {target_pop_size} graphs.')

        # Reset mutation probabilities to default
        self.mutation.update_requirements(requirements=self.requirements)
        return extended_pop
