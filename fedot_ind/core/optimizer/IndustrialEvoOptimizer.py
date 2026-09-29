from random import choice
from typing import Sequence, Optional, Dict, Any

from golem.core.dag.graph import Graph
from golem.core.optimisers.adaptive.mab_agents.contextual_mab_agent import ContextualMultiArmedBanditAgent
from golem.core.optimisers.adaptive.mab_agents.mab_agent import MultiArmedBanditAgent
from golem.core.optimisers.adaptive.mab_agents.neural_contextual_mab_agent import NeuralContextualMultiArmedBanditAgent
from golem.core.optimisers.adaptive.operator_agent import RandomAgent
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.operator import EvaluationOperator, PopulationT
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import Objective, ObjectiveFunction
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.populational_optimizer import _try_unfit_graph
from pymonad.either import Either

from fedot_ind.core.repository.IndustrialDispatcher import IndustrialDispatcher
from fedot_ind.core.repository.constanst_repository import FEDOT_MUTATION_STRATEGY
from fedot_ind.core.optimizer.observability import (
    CandidateRejectionReason,
    CandidateStatus,
    EvolutionDiagnosticsRecorder,
    EvolutionDiagnosticsSnapshot,
    build_evolution_summary,
    fitness_scalar,
    graph_identity,
    write_evolution_diagnostics,
)


class IndustrialPopulationError(RuntimeError):
    """Raised when no valid individual can seed Industrial optimisation."""

    def __init__(self, message: str, *, timed_out: bool = False):
        super().__init__(message)
        self.timed_out = timed_out


class IndustrialEvoOptimizer(EvoGraphOptimizer):
    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[OptGraph],
                 requirements: GraphRequirements,
                 graph_generation_params: GraphGenerationParams,
                 graph_optimizer_params: GPAlgorithmParameters,
                 optimisation_params: Optional[Dict[str, Any]] = None,
                 diagnostics_recorder: Optional[EvolutionDiagnosticsRecorder] = None):
        optimisation_params = dict(optimisation_params or {
            'mutation_agent': 'random',
            'mutation_strategy': 'params_mutation_strategy',
        })
        self.initial_graphs_prevalidated = bool(
            optimisation_params.get('initial_graphs_prevalidated', False)
        )
        self.diagnostics_output_dir = optimisation_params.get(
            'diagnostics_output_dir')
        self.diagnostics_recorder = diagnostics_recorder
        if self.diagnostics_recorder is None and self.diagnostics_output_dir:
            self.diagnostics_recorder = EvolutionDiagnosticsRecorder()
        graph_optimizer_params = self._init_industrial_optimizer_params(
            graph_optimizer_params, optimisation_params)
        super().__init__(objective, initial_graphs, requirements,
                         graph_generation_params, graph_optimizer_params)
        # self.operators.remove(self.crossover)
        self.evaluated_population = []
        self.requirements = requirements
        self.initial_graphs = initial_graphs
        self.eval_dispatcher = IndustrialDispatcher(
            adapter=graph_generation_params.adapter,
            n_jobs=requirements.n_jobs,
            graph_cleanup_fn=_try_unfit_graph,
            delegate_evaluator=graph_generation_params.remote_evaluator,
            diagnostics_recorder=self.diagnostics_recorder)

    @property
    def diagnostics(self) -> EvolutionDiagnosticsSnapshot:
        """Return an immutable snapshot of the current optimisation trace."""
        if self.diagnostics_recorder is None:
            return EvolutionDiagnosticsSnapshot(
                observations=(),
                summary=build_evolution_summary(()),
            )
        return self.diagnostics_recorder.snapshot()

    def export_diagnostics(self, output_dir):
        """Persist the current optimisation trace as JSONL, JSON and Markdown."""
        return write_evolution_diagnostics(self.diagnostics, output_dir)

    def _diagnostic_generation(self) -> Optional[int]:
        try:
            generation = self.current_generation_num
        except (AttributeError, TypeError):
            generation = None
        return int(generation) if generation is not None else None

    def _record_diagnostic(self, method_name: str, **values) -> None:
        recorder = self.diagnostics_recorder
        if recorder is not None:
            getattr(recorder, method_name)(**values)

    def _init_industrial_optimizer_params(self, graph_optimizer_params, optimisation_params):
        self.mutation_agent_dict = {'random': RandomAgent,
                                    'bandit': MultiArmedBanditAgent,
                                    'contextual_bandit': ContextualMultiArmedBanditAgent,
                                    'neural_bandit': NeuralContextualMultiArmedBanditAgent}
        # Min pop size to avoid getting stuck in local maximum during optimization.
        self.min_pop_size = 10
        # Min reproduce attempt for evolve and mutation stage.
        self.min_reproduce_attempt = 50
        # Max number of evaluations attempts to create graph for next pop
        self.graph_generation_attempts = 100
        graph_optimizer_params = self._exclude_resample_from_mutations(graph_optimizer_params)
        graph_optimizer_params.adaptive_mutation_type = self._set_optimisation_strategy(graph_optimizer_params,
                                                                                        optimisation_params)
        return graph_optimizer_params

    def _set_optimisation_strategy(self, graph_optimizer_params, optimisation_params):
        self.optimisation_mutation_probs = FEDOT_MUTATION_STRATEGY[optimisation_params['mutation_strategy']]
        mutation_agent = self.mutation_agent_dict[optimisation_params['mutation_agent']]
        if optimisation_params['mutation_agent'].__contains__('random'):
            mutation_agent = mutation_agent(actions=graph_optimizer_params.mutation_types,
                                            probs=self.optimisation_mutation_probs)
        else:
            mutation_agent = mutation_agent(actions=graph_optimizer_params.mutation_types)
        return mutation_agent

    def _exclude_resample_from_mutations(self, graph_optimizer_params):
        for mutation in graph_optimizer_params.mutation_types:
            try:
                is_invalid = mutation.__name__.__contains__('resample')
            except Exception:
                is_invalid = mutation.name.__contains__('resample')
            if is_invalid:
                graph_optimizer_params.mutation_types.remove(mutation)
        return graph_optimizer_params

    def _initial_population(self, evaluator: EvaluationOperator):
        """ Initializes the initial population """
        # Adding of initial assumptions to history as zero generation
        pop_size = self.graph_optimizer_params.pop_size
        label = 'initial_assumptions'
        initial_individuals = [Individual(graph, metadata=self.requirements.static_individual_metadata)
                               for graph in self.initial_graphs]

        if len(initial_individuals) < pop_size:  # in case we have only one init assumption
            # change strategy of init assumption creation. Set max probability to node change mutation
            self.mutation.agent._probs = FEDOT_MUTATION_STRATEGY['initial_population_diversity_strategy']
            initial_individuals = self._extend_population(initial_individuals, pop_size)
            self.mutation.agent._probs = self.optimisation_mutation_probs
            label = 'extended_initial_assumptions'
        init_population = evaluator(initial_individuals)
        if not init_population:
            self.log.warning(
                'Initial population was not evaluated within the shared time limit; '
                'retrying once without the timer guard.'
            )
            init_population = self.eval_dispatcher.evaluate_population(initial_individuals)
        if not init_population:
            raise IndustrialPopulationError(
                'Industrial optimisation cannot start because every initial graph failed evaluation.',
                timed_out=self.timer.is_time_limit_reached(),
            )
        self._update_population(next_population=init_population, evaluator=evaluator, label=label)
        return init_population, evaluator

    def _extend_population(self, pop: PopulationT, target_pop_size: int, mutation_prob: list = None) -> PopulationT:
        verifier, new_population, new_ind = self.graph_generation_params.verifier, list(pop), 'empty'
        pop_graphs = [ind.graph for ind in new_population]
        for iter_num in range(self.graph_generation_attempts):
            new_ind = 'empty'
            for repr_attempt in range(self.min_reproduce_attempt):
                random_ind = choice(pop)
                self._record_diagnostic(
                    'record_mutation_attempt',
                    generation=self._diagnostic_generation(),
                    individual_id=str(getattr(random_ind, 'uid', '')) or None,
                    graph_id=graph_identity(random_ind.graph),
                )
                new_ind = self.mutation(random_ind)
                if isinstance(new_ind, Individual):
                    # self.log.message(f'Successful mutation at attempt number: {repr_attempt}. '
                    #                  f'Obtain new pipeline - {new_ind.graph.descriptive_id}')
                    break
            if not isinstance(new_ind, Individual):
                self._record_diagnostic(
                    'record_candidate',
                    generation=self._diagnostic_generation(),
                    individual_id=None,
                    graph_id=None,
                    status=CandidateStatus.REJECTED,
                    reason=CandidateRejectionReason.MUTATION_DID_NOT_RETURN_INDIVIDUAL,
                )
            try:
                is_valid_graph = verifier(new_ind.graph)
            except Exception as error:
                self._record_diagnostic(
                    'record_candidate',
                    generation=self._diagnostic_generation(),
                    individual_id=str(getattr(new_ind, 'uid', '')) or None,
                    graph_id=graph_identity(
                        getattr(new_ind, 'graph', new_ind)),
                    status=CandidateStatus.REJECTED,
                    reason=CandidateRejectionReason.VERIFIER_FAILED,
                    error=error,
                )
                raise
            is_new_graph = new_ind.graph not in pop_graphs
            if all([is_new_graph, is_valid_graph]):
                new_population.append(new_ind)
                pop_graphs.append(new_ind.graph)
                self._record_diagnostic(
                    'record_candidate',
                    generation=self._diagnostic_generation(),
                    individual_id=str(new_ind.uid),
                    graph_id=graph_identity(new_ind.graph),
                    status=CandidateStatus.ACCEPTED,
                )
            else:
                reason = (
                    CandidateRejectionReason.VERIFIER_REJECTED
                    if not is_valid_graph
                    else CandidateRejectionReason.DUPLICATE_GRAPH
                )
                self._record_diagnostic(
                    'record_candidate',
                    generation=self._diagnostic_generation(),
                    individual_id=str(new_ind.uid),
                    graph_id=graph_identity(new_ind.graph),
                    status=CandidateStatus.REJECTED,
                    reason=reason,
                )
            if len(new_population) == target_pop_size:
                break
        return new_population

    def _update_population(self,
                           next_population: PopulationT,
                           evaluator: EvaluationOperator = None,
                           label: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        self.generations.append(next_population)
        if self.requirements.keep_history:
            self._log_to_history(next_population, label, metadata)
        self._iteration_callback(next_population, self)
        self.population = next_population
        graph_ids = tuple(graph_identity(individual.graph)
                          for individual in next_population)
        best_individuals = tuple(
            getattr(self.generations, 'best_individuals', ()))
        fitness_values = tuple(
            value for value in (
                fitness_scalar(getattr(individual, 'fitness', None))
                for individual in best_individuals
            ) if value is not None
        )
        self._record_diagnostic(
            'record_generation',
            generation=self._diagnostic_generation(),
            graph_ids=graph_ids,
            best_fitness=fitness_values[0] if fitness_values else None,
            label=label,
        )
        self.log.info(f'Generation num: {self.current_generation_num} size: {len(next_population)}')
        self.log.info(f'Best individuals: {str(self.generations)}')
        if self.generations.stagnation_iter_count > 0:
            self.log.info(f'no improvements for {self.generations.stagnation_iter_count} iterations')
            self.log.info(f'spent time: {round(self.timer.minutes_from_start, 1)} min')
        return next_population

    def _evolve_population(self,
                           population: PopulationT,
                           evaluator: EvaluationOperator) -> PopulationT:
        """ Method realizing full evolution cycle """

        def evolve_pop(population, evaluator):
            individuals_to_select = self.regularization(population, evaluator)
            new_population = self.reproducer.reproduce(individuals_to_select, evaluator)
            if self.reproducer.stop_condition or new_population is None:
                new_population = population
            else:
                self.log.message(f'Successful reproduction')

            # Adaptive agent experience collection & learning
            # Must be called after reproduction (that collects the new experience)
            experience = self.mutation.agent_experience
            experience.collect_results(new_population)
            self.mutation.agent.partial_fit(experience)

            # Use some part of previous pop in the next pop
            new_population = self.inheritance(population, new_population)
            new_population = self.elitism(self.generations.best_individuals, new_population)
            return new_population, evaluator

        return evolve_pop(population, evaluator)

    def get_structure_unique_population(self, population: PopulationT, evaluator: EvaluationOperator) -> PopulationT:
        """ Increases structurally uniqueness of population to prevent stagnation in optimization process.
        Returned population may be not entirely unique, if the size of unique population is lower than MIN_POP_SIZE. """
        unique_population_with_ids = {ind.graph.descriptive_id: ind for ind in population}
        unique_population = list(unique_population_with_ids.values())
        is_population_too_small = len(unique_population) < self.min_pop_size
        # if size of unique population is too small, then extend it to MIN_POP_SIZE by repeating individuals
        if all([is_population_too_small, not self.reproducer.stop_condition]):
            self.mutation.agent._probs = FEDOT_MUTATION_STRATEGY['unique_population_strategy']
            unique_population = self._extend_population(pop=unique_population,
                                                        target_pop_size=self.min_pop_size)
            self.mutation.agent._probs = self.optimisation_mutation_probs
            population = evaluator(unique_population)
        return population, evaluator

    def _update_requirements(self, population, evaluator):
        # Defines adaptive changes to algorithm parameters like pop_size and operator probabilities
        if not self.generations.is_any_improved:
            self.graph_optimizer_params.mutation_prob, self.graph_optimizer_params.crossover_prob = \
                self._operators_prob.next(population)
            self.log.info(
                f'Next mutation proba: {self.graph_optimizer_params.mutation_prob}; '
                f'Next crossover proba: {self.graph_optimizer_params.crossover_prob}')
        self.graph_optimizer_params.pop_size = self._pop_size.next(population)
        self.requirements.max_depth = self._graph_depth.next()
        self.log.info(f'Next population size: {self.graph_optimizer_params.pop_size}; '
                      f''f'max graph depth: {self.requirements.max_depth}')

        # update requirements in operators
        for operator in self.operators:
            operator.update_requirements(self.graph_optimizer_params, self.requirements)
        return population, evaluator

    def _optimise_loop(self, population_to_eval, evaluator):
        evaluated_population = Either.insert((population_to_eval, evaluator)). \
            then(lambda opt_data: self._update_requirements(*opt_data)). \
            then(lambda opt_data: self._evolve_population(*opt_data)). \
            then(lambda fitness_data: self.get_structure_unique_population(*fitness_data)). \
            then(lambda reg_data: self._update_population(*reg_data)).value
        return evaluated_population

    def optimise(self, objective: ObjectiveFunction) -> Sequence[Graph]:
        """Optimise graphs and always persist diagnostics when configured."""
        try:
            return self._optimise(objective)
        finally:
            output_dir = getattr(self, 'diagnostics_output_dir', None)
            if output_dir:
                self.export_diagnostics(output_dir)

    def _optimise(self, objective: ObjectiveFunction) -> Sequence[Graph]:
        with self.timer:
            pbar = self._progressbar
            try:
                evaluator = self.eval_dispatcher.dispatch(objective, self.timer)
                try:
                    population_to_eval, evaluator = self._initial_population(evaluator)
                except IndustrialPopulationError as error:
                    if error.timed_out and self.initial_graphs_prevalidated and self.initial_graphs:
                        self.log.warning(
                            'Industrial composition exhausted its time budget after FEDOT had '
                            'prevalidated the initial assumption; returning that assumption.'
                        )
                        return list(self.initial_graphs[:1])
                    raise
                self.evaluated_population.append(population_to_eval)
                while not self.stop_optimization():
                    population_to_eval = self._optimise_loop(population_to_eval=population_to_eval,
                                                             evaluator=evaluator)
                    self.evaluated_population.append(population_to_eval)
                    pbar.update()
            finally:
                pbar.close()
        self._update_population(self.best_individuals, None, 'final_choices')
        best_models = [ind.graph for ind in self.best_individuals]
        return best_models
