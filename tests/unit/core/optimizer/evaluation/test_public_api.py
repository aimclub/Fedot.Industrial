"""Small public API runs; these are correctness checks, not timing evidence."""

from copy import deepcopy
import json

import numpy as np
import torch
import cv2  # noqa: F401
import pytest
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.core.optimizer.configuration import (
    EvolutionConfig, ExecutionPolicy, MutationPlanningPolicy, ResourceBudget,
)
from fedot_ind.core.optimizer.evaluation.configuration import EvaluationBackend, EvaluationExecutionConfig
from fedot_ind.core.optimizer.evaluation.dispatcher import LocalEvaluationDispatcher
from fedot_ind.core.repository.config_repository import (
    DEFAULT_CLF_API_CONFIG, DEFAULT_REG_API_CONFIG, DEFAULT_TSF_API_CONFIG,
)
from fedot_ind.integration.fedot.extensions import industrial_extension_scope


def _inputs(task):
    if task == 'ts_forecasting':
        history = np.sin(np.arange(80) / 5.) + np.arange(80) / 20.
        return (history, history), (history, None), 3
    generator = np.random.default_rng(42)
    features = generator.normal(size=(32, 1, 24)).astype(np.float32)
    target = (features.mean(axis=(1, 2)) > 0).astype(int)
    if task == 'regression':
        target = 2. * features.mean(axis=(1, 2)) + features.std(axis=(1, 2))
    return (features, target), (features[:5].copy(), None), 5


def _config(task, backend, root):
    templates = {'classification': DEFAULT_CLF_API_CONFIG, 'regression': DEFAULT_REG_API_CONFIG,
                 'ts_forecasting': DEFAULT_TSF_API_CONFIG}
    config = deepcopy(templates[task])
    diagnostics = root / 'diagnostics'
    policy = EvolutionConfig(
        random_seed=42, mutation_planning=MutationPlanningPolicy(enabled=True),
        resource_budget=ResourceBudget(2, 8, 8),
        execution_policy=ExecutionPolicy(retry_initial_population_without_timer=False,
                                         diagnostics_output_dir=str(diagnostics)),
        evaluation_execution=EvaluationExecutionConfig(backend, candidate_timeout_seconds=90,
                                                       total_timeout_seconds=180))
    config['automl_config']['optimisation_strategy'] = {
        'optimisation_agent': 'Industrial', 'optimisation_strategy': policy.to_record()}
    with industrial_extension_scope():
        if task == 'ts_forecasting':
            builders = [PipelineBuilder().add_node('hankelisation').add_node('ridge'),
                        PipelineBuilder().add_node('gaussian_filter').add_node('hankelisation').add_node('ridge')]
            operations = ['hankelisation', 'ridge', 'gaussian_filter']
        else:
            model = 'pdl_clf' if task == 'classification' else 'pdl_reg'
            downstream = 'dt' if task == 'classification' else 'dtreg'
            parameters = {'model': downstream, 'max_pairs': 128, 'random_state': 42}
            features = {'use_cache': False, 'use_sliding_window': False, 'window_size': 0}
            builders = [PipelineBuilder().add_node(name, params=features).add_node(model, params=parameters)
                        for name in ('quantile_extractor', 'quantile_extractor_torch')]
            operations = [model, 'quantile_extractor', 'quantile_extractor_torch', downstream]
        config['automl_config']['initial_assumption'] = [builder.build() for builder in builders]
        config['industrial_config']['initial_assumption'] = config['automl_config']['initial_assumption']
    config['automl_config']['available_operations'] = operations
    config['learning_config']['learning_strategy_params'] = {
        'timeout': 3, 'pop_size': 2, 'num_of_generations': 1, 'n_jobs': 1, 'with_tuning': False,
        'seed': 42, 'max_depth': 3, 'logging_level': 50,
        'use_operations_cache': False, 'use_predictions_cache': False,
        'cv_folds': None if task == 'ts_forecasting' else 2,
    }
    config['learning_config']['optimisation_loss'] = {
        'quality_loss': 'f1' if task == 'classification' else 'rmse'}
    if task == 'ts_forecasting':
        config['automl_config']['task_params'] = {'forecast_length': 3}
        config['industrial_config']['task_params'] = {'forecast_length': 3}
    config['compute_config'].update(
        backend='cpu', distributed={'processes': False, 'n_workers': 1, 'threads_per_worker': 1, 'memory_limit': 0.3},
        output_folder=str(root / 'runtime'),
        automl_folder={'optimisation_history': str(root / 'history'), 'composition_results': str(root / 'models')})
    return config


@pytest.mark.parametrize('backend', [EvaluationBackend.SEQUENTIAL, EvaluationBackend.PROCESS])
@pytest.mark.parametrize('task', ['classification', 'regression', 'ts_forecasting'])
def test_explicit_executor_runs_through_fedot_industrial_fit_predict(task, backend, tmp_path, monkeypatch):
    from fedot.extensions import get_registered_extensions
    previous = tuple(get_registered_extensions())
    training, prediction, expected_samples = _inputs(task)
    original = tuple(None if value is None else value.copy() for value in training)
    observed_backends = []
    original_init = LocalEvaluationDispatcher.__init__

    def observe_dispatcher(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        observed_backends.append(self.config.backend)

    monkeypatch.setattr(LocalEvaluationDispatcher, '__init__', observe_dispatcher)
    model = FedotIndustrial(**_config(task, backend, tmp_path))
    try:
        model.fit(training)
        values = model.predict(prediction)
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        values = np.asarray(values)
        assert values.size == expected_samples
        assert np.isfinite(values).all()
        if task == 'classification':
            assert set(values.reshape(-1)).issubset(set(training[1]))
        summary = json.loads((tmp_path / 'diagnostics/evolution_summary.json').read_text(encoding='utf-8'))
        assert summary['evaluations_succeeded'] >= 2
        assert summary['generations_observed'] >= 2
        events = [json.loads(line) for line in (tmp_path / 'diagnostics/evolution_events.jsonl').read_text(
            encoding='utf-8').splitlines()]
        initial = next(event for event in events if event.get('label') == 'initial_assumptions')
        evolved = [event for event in events if event['event_type'] == 'generation'
                   and event.get('generation', 0) > initial['generation'] and not event.get('label')]
        assert evolved
        assert any(event['event_type'] == 'evaluation' and event.get('status') == 'succeeded'
                   and initial['sequence'] < event['sequence'] < evolved[0]['sequence'] for event in events)
        # The public configuration must reach the actual dispatcher, not a fallback.
        assert observed_backends == [backend]
    finally:
        model.shutdown()
    for current, snapshot in zip(training, original):
        np.testing.assert_array_equal(current, snapshot)
    assert tuple(get_registered_extensions()) == previous
