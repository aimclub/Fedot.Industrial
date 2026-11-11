"""Protocol boundary used by the existing measured search interpreter."""
from dataclasses import dataclass, replace
import json
from pathlib import Path
from time import perf_counter

from .data import load_training_data


def load_protocol():
    return json.loads(Path(__file__).with_name('protocol.json').read_text('utf-8'))


@dataclass(frozen=True)
class MutationStudy:
    arm: str
    task: str

    @property
    def strategy(self):
        from fedot_ind.core.optimizer.configuration import MutationStrategy
        return MutationStrategy(load_protocol()['arms'][self.arm])

    def load_data(self):
        return load_training_data(self.task, load_protocol())

    def search_settings(self, settings):
        return settings

    def configure_environment(self, params):
        return params

    def configure_evolution(self, config):
        return config

    def prepare_objective(self, measured, data):
        return measured

    def configure_selected(self, pipeline, parameters):
        from benchmark.industrial.experiments.mutation_planning.population_study.preparation import configure_pipeline
        return configure_pipeline(pipeline, parameters)

    def implementation_paths(self):
        import importlib
        optimizer_module = importlib.import_module('fedot_ind.core.optimizer.IndustrialEvoOptimizer')
        directory = Path(optimizer_module.__file__).parent
        return [*Path(__file__).parent.glob('*.py'), Path(__file__).with_name('protocol.json'),
                directory / 'mutation_profiles.json', directory.parent / 'repository/constanst_repository.py']

    def preparation(self, base, root):
        return AuditedPreparation(base, str(root / 'preparation.jsonl'))

    def observe_mutations(self, optimizer, root):
        from fedot_ind.core.optimizer.randomness import SeededOperator
        mutation = optimizer.mutation
        while isinstance(mutation, SeededOperator):
            mutation = mutation.operator
        previous = mutation.observer

        def observe(execution):
            if previous is not None:
                previous(execution)
            record = execution.to_record()
            from fedot_ind.core.optimizer.mutation import mutation_name
            record['agent_probabilities'] = dict(zip(
                map(mutation_name, mutation.operator.agent.available_actions),
                mutation.operator.agent.get_action_probs(None)))
            with (root / 'mutations.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')

        mutation.observer = observe

    def evaluate_selected(self, winner, data, root):
        import pandas as pd
        from benchmark.industrial.experiments.evolution_parallelism.final_evaluation import FinalEvaluation
        from benchmark.industrial.experiments.evolution_parallelism.search import loss_value, to_numpy
        try:
            start = perf_counter()
            winner.fit(data.refit)
            fit_seconds = perf_counter() - start
            start = perf_counter()
            output = winner.predict(replace(data.test, target=None, state='predict'),
                                    output_mode='labels' if self.task == 'classification' else 'default')
            prediction_seconds = perf_counter() - start
            loss = loss_value(self.task, data.test.target, output.predict)
            pd.DataFrame(dict(target=to_numpy(data.test.target).reshape(-1),
                              prediction=to_numpy(output.predict).reshape(-1))).to_csv(
                                  root / 'outer_train_predictions.csv', index=False)
            return FinalEvaluation(fit_seconds, prediction_seconds, None), loss
        finally:
            winner.unfit()


@dataclass(frozen=True)
class AuditedPreparation:
    base: object
    trace_path: str

    def __call__(self, graph):
        from copy import deepcopy
        from hashlib import sha256
        from benchmark.industrial.experiments.mutation_planning.population_study.preparation import pipeline_configuration
        from fedot_ind.core.optimizer.evaluation.preparation import canonical_json
        from fedot_ind.integration.fedot.extensions import industrial_extension_scope
        with industrial_extension_scope():
            pipeline = self.base.adapter.restore(deepcopy(graph))
            before = canonical_json(pipeline_configuration(pipeline, self.base.task))
        prepared = self.base(graph)
        payload = json.loads(prepared.configuration_json)
        typed = [{'operation': node['operation'], 'parents': node['parents']} for node in payload['nodes']]
        topology = [node['parents'] for node in payload['nodes']]
        record = dict(before_id=sha256(before.encode()).hexdigest(), configuration_id=prepared.configuration_id,
                      structure_id=sha256(canonical_json(typed).encode()).hexdigest(),
                      topology_id=sha256(canonical_json(topology).encode()).hexdigest(), nodes=len(typed))
        with Path(self.trace_path).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        return prepared
