from random import choice, Random
from copy import deepcopy
from contextlib import contextmanager
from typing import Any, Dict, Mapping, Optional, Sequence

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
from fedot_ind.core.optimizer.evaluation.configuration import EvaluationBackend
from fedot_ind.core.optimizer.evaluation.dispatcher import LocalEvaluationDispatcher
from fedot_ind.core.optimizer.evaluation.preparation import GraphPreparation
from fedot_ind.core.optimizer.evaluation.archive import CompletedEvaluationArchive, record_final_history
from fedot_ind.core.repository.constanst_repository import FEDOT_MUTATION_STRATEGY
from fedot_ind.core.optimizer.configuration import (
    EvolutionConfig,
    MutationAgentType,
    normalize_evolution_config,
)
from fedot_ind.core.optimizer.domain import (
    CandidateFailure,
    CandidateFailureCode,
    EvolutionPhase,
    EvolutionState,
    IndustrialPopulationError,
    classify_candidate_failure,
    transition_evolution_state,
)
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
from fedot_ind.core.optimizer.mutation import mutation_weights, without_resample_mutations
from fedot_ind.core.optimizer.randomness import (
    CandidateRandomness, SeededOperator, install_seeded_operators, operator_random_scope,
)
from fedot_ind.core.optimizer.mutation_runtime import (
    MutationExecutionStatus, PlannedMutation, install_extension_aware_advisor, install_planned_mutation,
)
from fedot_ind.integration.fedot.evaluation import confirmed_input_from_objective
from fedot_ind.core.optimizer.graph_validation import (
    IndustrialGraphVerifier,
    ModelFeatureFlow,
    infer_validation_context,
)


EXPECTED_VERIFIER_EXCEPTIONS = (
    AttributeError,
    IndexError,
    KeyError,
    TypeError,
    ValueError,
)


class IndustrialEvoOptimizer(EvoGraphOptimizer):
    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[OptGraph],
                 requirements: GraphRequirements,
                 graph_generation_params: GraphGenerationParams,
                 graph_optimizer_params: GPAlgorithmParameters,
                 optimisation_params: Optional[Mapping[str, Any] | EvolutionConfig] = None,
                 diagnostics_recorder: Optional[EvolutionDiagnosticsRecorder] = None,
                 evaluation_graph_preparation: GraphPreparation | None = None):
        self.evolution_config = normalize_evolution_config(optimisation_params)
        execution_policy = self.evolution_config.execution_policy
        self.initial_graphs_prevalidated = execution_policy.initial_graphs_prevalidated
        self.retry_initial_population_without_timer = execution_policy.retry_initial_population_without_timer
        self.diagnostics_output_dir = execution_policy.diagnostics_output_dir
        self.evolution_state = EvolutionState.created()
        self.diagnostics_recorder = diagnostics_recorder
        if self.diagnostics_recorder is None and self.diagnostics_output_dir:
            self.diagnostics_recorder = EvolutionDiagnosticsRecorder()
        graph_optimizer_params = self._init_industrial_optimizer_params(
            graph_optimizer_params, self.evolution_config)
        install_extension_aware_advisor(graph_generation_params)
        legacy_verifier = graph_generation_params.verifier
        task_type, requested_device = infer_validation_context(
            graph_generation_params)
        if isinstance(legacy_verifier, IndustrialGraphVerifier):
            self.graph_verifier = legacy_verifier
        else:
            self.graph_verifier = IndustrialGraphVerifier(
                adapter=graph_generation_params.adapter,
                task_type=task_type,
                requested_device=requested_device,
                legacy_verifier=legacy_verifier,
            )
            graph_generation_params.verifier = self.graph_verifier
        super().__init__(objective, initial_graphs, requirements,
                         graph_generation_params, graph_optimizer_params)
        install_planned_mutation(self, self.evolution_config.mutation_planning)
        install_seeded_operators(self, self.evolution_config.random_seed)
        self._parent_selection_attempt = 0
        self._agent_learning_attempt = 0
        # self.operators.remove(self.crossover)
        self.evaluated_population = []
        self.requirements = requirements
        self.completed_evaluations = CompletedEvaluationArchive(objective, requirements.keep_n_best)
        self._final_individuals = ()
        execution_config = self.evolution_config.evaluation_execution
        dispatcher_type = (IndustrialDispatcher if execution_config.backend is EvaluationBackend.LEGACY
                           else LocalEvaluationDispatcher)
        if dispatcher_type is IndustrialDispatcher and evaluation_graph_preparation is not None:
            raise ValueError('Explicit graph preparation requires a local evaluation backend')
        execution_args = ({} if dispatcher_type is IndustrialDispatcher else
                          {"config": execution_config, "random_seed": self.evolution_config.random_seed,
                           "graph_preparation": evaluation_graph_preparation})
        self.eval_dispatcher = dispatcher_type(
            adapter=graph_generation_params.adapter,
            n_jobs=requirements.n_jobs,
            graph_cleanup_fn=_try_unfit_graph,
            delegate_evaluator=graph_generation_params.remote_evaluator,
            completed_evaluation_callback=self.completed_evaluations.record_results,
            diagnostics_recorder=self.diagnostics_recorder, **execution_args)

    @property
    def final_individuals(self):
        """Detached final choices; empty until search has finished."""
        return getattr(self, '_final_individuals', ())

    @property
    def best_individuals(self):
        if self.evolution_state.phase is EvolutionPhase.COMPLETED and self.final_individuals:
            return self.final_individuals
        return super().best_individuals

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

    def _record_mutation_execution(self, execution) -> None:
        self._record_diagnostic(
            'record_mutation_plan', generation=self._diagnostic_generation(),
            label=execution.action, status=execution.status.value,
            plans_examined=execution.examined, admissible_plans=execution.admissible,
            repair_edits=execution.repair_edits, validation_issue_codes=execution.issue_codes,
        )

    def _transition_evolution(
            self,
            target: EvolutionPhase,
            *,
            generation: int | None = None,
    ) -> EvolutionState:
        current = getattr(self, 'evolution_state', EvolutionState.created())
        self.evolution_state = transition_evolution_state(
            current,
            target,
            generation=generation,
        )
        return self.evolution_state

    def _init_industrial_optimizer_params(
            self,
            graph_optimizer_params,
            config: EvolutionConfig,
    ):
        self.mutation_agent_dict = {
            MutationAgentType.RANDOM: RandomAgent,
            MutationAgentType.BANDIT: MultiArmedBanditAgent,
            MutationAgentType.CONTEXTUAL_BANDIT: ContextualMultiArmedBanditAgent,
            MutationAgentType.NEURAL_BANDIT: NeuralContextualMultiArmedBanditAgent,
        }
        resource_budget = config.resource_budget
        # Min pop size to avoid getting stuck in local maximum during optimization.
        self.min_pop_size = resource_budget.min_population_size
        # Min reproduce attempt for evolve and mutation stage.
        self.min_reproduce_attempt = resource_budget.mutation_attempts_per_candidate
        # Max number of evaluations attempts to create graph for next pop
        self.graph_generation_attempts = resource_budget.population_extension_attempts
        graph_optimizer_params = self._exclude_resample_from_mutations(graph_optimizer_params)
        graph_optimizer_params.adaptive_mutation_type = self._set_optimisation_strategy(graph_optimizer_params,
                                                                                        config)
        return graph_optimizer_params

    def _set_optimisation_strategy(self, graph_optimizer_params, config: EvolutionConfig):
        self.optimisation_mutation_probs = mutation_weights(
            graph_optimizer_params.mutation_types, FEDOT_MUTATION_STRATEGY[config.mutation_strategy.value])
        mutation_agent = self.mutation_agent_dict[config.mutation_agent]
        if config.mutation_agent is MutationAgentType.RANDOM:
            mutation_agent = mutation_agent(actions=graph_optimizer_params.mutation_types,
                                            probs=self.optimisation_mutation_probs)
        else:
            mutation_agent = mutation_agent(actions=graph_optimizer_params.mutation_types)
        return mutation_agent

    def _exclude_resample_from_mutations(self, graph_optimizer_params):
        graph_optimizer_params.mutation_types = without_resample_mutations(
            graph_optimizer_params.mutation_types
        )
        return graph_optimizer_params

    @contextmanager
    def _mutation_probability_strategy(self, strategy):
        agent = self.mutation.agent
        if not isinstance(agent, RandomAgent):
            yield
            return
        previous = agent._probs
        agent._probs = mutation_weights(agent.available_actions, FEDOT_MUTATION_STRATEGY[strategy])
        try:
            yield
        finally:
            agent._probs = previous

    def _initial_population(self, evaluator: EvaluationOperator):
        """ Initializes the initial population """
        # Adding of initial assumptions to history as zero generation
        pop_size = self.graph_optimizer_params.pop_size
        label = 'initial_assumptions'
        graphs = self._admissible_initial_graphs()
        if not graphs:
            raise IndustrialPopulationError(
                'No initial graph satisfies the confirmed input contract.', timed_out=False)
        initial_individuals = [Individual(graph, metadata=deepcopy(self.requirements.static_individual_metadata))
                               for graph in graphs]

        if len(initial_individuals) < pop_size:  # in case we have only one init assumption
            # change strategy of init assumption creation. Set max probability to node change mutation
            with self._mutation_probability_strategy('initial_population_diversity_strategy'):
                initial_individuals = self._extend_population(initial_individuals, pop_size)
            label = 'extended_initial_assumptions'
        init_population = evaluator(initial_individuals)
        if (not init_population and self.retry_initial_population_without_timer
                and getattr(self.eval_dispatcher, 'allows_untimed_retry', True)):
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

    def _admissible_initial_graphs(self):
        verifier = getattr(self, 'graph_verifier', None)
        if not isinstance(verifier, IndustrialGraphVerifier):
            return list(self.initial_graphs)
        graphs = []
        for graph in self.initial_graphs:
            report = verifier.verify_with_report(graph)
            if report.is_valid:
                graphs.append(graph)
            else:
                self._record_candidate_failure(CandidateFailure(
                    code=CandidateFailureCode.VERIFIER_REJECTED,
                    message='Initial graph violates the confirmed input or graph contract.',
                    graph_id=graph_identity(graph), validation_issue_codes=report.error_codes,
                ))
        return graphs

    def _extend_population(self, pop: PopulationT, target_pop_size: int, mutation_prob: list = None) -> PopulationT:
        del mutation_prob
        verifier, new_population = self.graph_generation_params.verifier, list(pop)
        if len(new_population) >= target_pop_size:
            return new_population
        if not pop:
            raise IndustrialPopulationError('Cannot extend an empty population.')
        run_seed = getattr(self, 'evolution_config', EvolutionConfig()).random_seed
        pop_graphs = [ind.graph for ind in new_population]
        exhausted_parents = set()
        timer = getattr(self, 'timer', None)
        for _ in range(self.graph_generation_attempts):
            available_parents = [parent for parent in pop if id(parent) not in exhausted_parents]
            if not available_parents or (timer is not None and timer.is_time_limit_reached()):
                break
            new_ind = None
            for _ in range(self.min_reproduce_attempt):
                if timer is not None and timer.is_time_limit_reached():
                    break
                if run_seed is None:
                    random_ind = choice(available_parents)
                else:
                    attempt = self._parent_selection_attempt
                    self._parent_selection_attempt += 1
                    seed = CandidateRandomness(run_seed, 'parent_selection', attempt).seed
                    random_ind = Random(seed).choice(available_parents)
                self._record_diagnostic(
                    'record_mutation_attempt',
                    generation=self._diagnostic_generation(),
                    individual_id=str(getattr(random_ind, 'uid', '')) or None,
                    graph_id=graph_identity(random_ind.graph),
                )
                new_ind = self.mutation(random_ind)
                if isinstance(new_ind, Individual):
                    break
                execution = getattr(self.mutation, 'last_execution', None)
                if execution is not None and execution.status is MutationExecutionStatus.NO_ADMISSIBLE_PLAN:
                    # Repeating RNG draws cannot change an empty deterministic plan pool.
                    exhausted_parents.add(id(random_ind))
                    break
            if not isinstance(new_ind, Individual):
                failure = classify_candidate_failure(
                    is_individual=False,
                    is_valid_graph=None,
                    is_duplicate_graph=None,
                )
                self._record_candidate_failure(failure)
                continue

            graph_id = graph_identity(new_ind.graph)
            individual_id = str(new_ind.uid)
            validation_issue_codes = ()
            try:
                if isinstance(verifier, IndustrialGraphVerifier):
                    validation_report = verifier.verify_with_report(new_ind.graph)
                    is_valid_graph = validation_report.is_valid
                    validation_issue_codes = validation_report.error_codes
                else:
                    is_valid_graph = verifier(new_ind.graph)
            except EXPECTED_VERIFIER_EXCEPTIONS as error:
                failure = CandidateFailure(
                    code=CandidateFailureCode.VERIFIER_FAILED,
                    message=str(error),
                    individual_id=individual_id,
                    graph_id=graph_id,
                )
                self._record_candidate_failure(failure, error=error)
                raise
            failure = classify_candidate_failure(
                is_individual=True,
                is_valid_graph=is_valid_graph,
                is_duplicate_graph=new_ind.graph in pop_graphs,
                individual_id=individual_id,
                graph_id=graph_id,
                validation_issue_codes=validation_issue_codes if not is_valid_graph else (),
            )
            if failure is None:
                new_population.append(new_ind)
                pop_graphs.append(new_ind.graph)
                self._record_diagnostic(
                    'record_candidate',
                    generation=self._diagnostic_generation(),
                    individual_id=individual_id,
                    graph_id=graph_id,
                    status=CandidateStatus.ACCEPTED,
                )
            else:
                self._record_candidate_failure(failure)
            if len(new_population) == target_pop_size:
                break
        return new_population

    def _record_candidate_failure(
            self,
            failure: CandidateFailure,
            *,
            error: BaseException | None = None,
    ) -> None:
        self._record_diagnostic(
            'record_candidate',
            generation=self._diagnostic_generation(),
            individual_id=failure.individual_id,
            graph_id=failure.graph_id,
            status=CandidateStatus.REJECTED,
            reason=CandidateRejectionReason(failure.code.value),
            error=error,
            validation_issue_codes=failure.validation_issue_codes,
        )

    def _update_population(self,
                           next_population: PopulationT,
                           evaluator: EvaluationOperator = None,
                           label: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        final = label == 'final_choices'
        if not final:
            self.generations.append(next_population)
        if self.requirements.keep_history:
            if final:
                record_final_history(self.history, next_population, metadata)
                if self.requirements.history_dir:
                    self.history.save_current_results(self.requirements.history_dir)
            else:
                self._log_to_history(next_population, label, metadata)
        self._iteration_callback(next_population, self)
        self.population = next_population
        graph_ids = tuple(graph_identity(individual.graph)
                          for individual in next_population)
        best_individuals = tuple(next_population if final else
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
        state = getattr(self, 'evolution_state', EvolutionState.created())
        generation = self._diagnostic_generation()
        if state.phase is EvolutionPhase.EVOLVING and generation is not None:
            self._transition_evolution(EvolutionPhase.EVOLVING, generation=generation)
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
            if new_population is None:
                new_population = population
            else:
                self.log.info('Successful reproduction')

            # Adaptive agent experience collection & learning
            # Must be called after reproduction (that collects the new experience)
            experience = self.mutation.agent_experience
            experience.collect_results(new_population)
            run_seed = self.evolution_config.random_seed
            if run_seed is None:
                self.mutation.agent.partial_fit(experience)
            else:
                address = CandidateRandomness(run_seed, 'agent_learning', self._agent_learning_attempt)
                self._agent_learning_attempt += 1
                with operator_random_scope(address.seed):
                    self.mutation.agent.partial_fit(experience)

            # Use some part of previous pop in the next pop
            new_population = self.inheritance(population, new_population)
            new_population = self.elitism(self.generations.best_individuals, new_population)
            return new_population, evaluator

        return evolve_pop(population, evaluator)

    def get_structure_unique_population(self, population: PopulationT, evaluator: EvaluationOperator) -> PopulationT:
        """ Increases structurally uniqueness of population to prevent stagnation in optimization process.
        Returned population may contain duplicates if its unique part is smaller than MIN_POP_SIZE.
        """
        unique_population_with_ids = {ind.graph.descriptive_id: ind for ind in population}
        unique_population = list(unique_population_with_ids.values())
        is_population_too_small = len(unique_population) < self.min_pop_size
        # if size of unique population is too small, then extend it to MIN_POP_SIZE by repeating individuals
        if is_population_too_small and not self.timer.is_time_limit_reached():
            with self._mutation_probability_strategy('unique_population_strategy'):
                unique_population = self._extend_population(pop=unique_population,
                                                            target_pop_size=self.min_pop_size)
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
        verifier = getattr(self, 'graph_verifier', None)
        previous_input_type = getattr(verifier, 'input_data_type', None)
        previous_feature_flow = getattr(verifier, 'model_feature_flow', None)
        mutation = getattr(self, 'mutation', None)
        while isinstance(mutation, SeededOperator):
            mutation = mutation.operator
        previous_raw_input = getattr(mutation, 'raw_input', None)
        try:
            raw_input = confirmed_input_from_objective(objective)
            if isinstance(verifier, IndustrialGraphVerifier) and raw_input is not None:
                verifier.input_data_type = raw_input.data_type
                verifier.model_feature_flow = ModelFeatureFlow.TENSOR_FEATURES
            if isinstance(mutation, PlannedMutation):
                mutation.raw_input = raw_input
            return self._optimise(objective)
        finally:
            if isinstance(verifier, IndustrialGraphVerifier):
                verifier.input_data_type = previous_input_type
                verifier.model_feature_flow = previous_feature_flow
            if isinstance(mutation, PlannedMutation):
                mutation.raw_input = previous_raw_input
            state = getattr(self, 'evolution_state', EvolutionState.created())
            if state.phase not in (EvolutionPhase.COMPLETED, EvolutionPhase.FAILED):
                self._transition_evolution(EvolutionPhase.FAILED)
            output_dir = getattr(self, 'diagnostics_output_dir', None)
            if output_dir:
                self.export_diagnostics(output_dir)

    def _optimise(self, objective: ObjectiveFunction) -> Sequence[Graph]:
        self._transition_evolution(EvolutionPhase.INITIALISING)
        with self.timer:
            pbar = self._progressbar
            try:
                evaluator = self.eval_dispatcher.dispatch(objective, self.timer)
                self._transition_evolution(EvolutionPhase.EVALUATING_INITIAL)
                try:
                    population_to_eval, evaluator = self._initial_population(evaluator)
                except IndustrialPopulationError as error:
                    if error.timed_out and self.initial_graphs_prevalidated and self.initial_graphs:
                        verifier = getattr(self, 'graph_verifier', None)
                        fallback = next((graph for graph in self.initial_graphs
                                         if not isinstance(verifier, IndustrialGraphVerifier)
                                         or verifier.verify(graph)),
                                        None)
                        if fallback is None:
                            raise
                        self.log.warning(
                            'Industrial composition exhausted its time budget after FEDOT had '
                            'prevalidated the initial assumption; returning that assumption.'
                        )
                        self._transition_evolution(EvolutionPhase.COMPLETED)
                        return [fallback]
                    raise
                # GOLEM predicts the next generation from evolution time, not initial evaluation.
                # Keep the original timer start: initialisation still consumes the shared budget.
                self.timer.set_init_time(self.timer.minutes_from_start)
                self._transition_evolution(EvolutionPhase.EVOLVING)
                self.evaluated_population.append(population_to_eval)
                while not self.stop_optimization():
                    population_to_eval = self._optimise_loop(population_to_eval=population_to_eval,
                                                             evaluator=evaluator)
                    self.evaluated_population.append(population_to_eval)
                    pbar.update()
            finally:
                pbar.close()
        archive = getattr(self, 'completed_evaluations', None)
        choices = () if archive is None else archive.best_individuals
        self._final_individuals = choices or tuple(self.best_individuals)
        self._update_population(self.final_individuals, None, 'final_choices')
        self._transition_evolution(EvolutionPhase.COMPLETED)
        best_models = [ind.graph for ind in self.best_individuals]
        return best_models
