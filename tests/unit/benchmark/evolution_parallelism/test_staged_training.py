from dataclasses import replace
import json
import sys

import pytest

from benchmark.industrial.experiments.staged_mutations.training import (
    StagedTrainingStudy, load_training_protocol, search_policy)
from benchmark.industrial.experiments.staged_mutations.training_runner import job_matrix


def test_matrix_has_eighteen_paired_jobs_and_deterministic_order():
    protocol = load_training_protocol()
    jobs = job_matrix(protocol)
    assert len(jobs) == len(set(jobs)) == 18
    assert jobs == job_matrix(protocol)
    for task in protocol['tasks']:
        for seed in protocol['seeds']:
            assert {mode for t, mode, s in jobs if t == task and s == seed} == {'fixed', 'staged'}
    assert len(job_matrix(protocol, pilot=True)) == 3


def test_both_arms_share_everything_except_the_schedule():
    from fedot_ind.core.optimizer.configuration import EvolutionConfig
    first = StagedTrainingStudy('fixed', 'regression').configure_evolution(EvolutionConfig())
    second = StagedTrainingStudy('staged', 'regression').configure_evolution(EvolutionConfig())
    assert replace(first, mutation_schedule=None) == replace(second, mutation_schedule=None)
    assert first.mutation_schedule.parameters == second.mutation_schedule.parameters
    assert len(first.mutation_schedule.stages) == 1 and len(second.mutation_schedule.stages) == 3
    assert {stage.family_fraction for stage in (
        *first.mutation_schedule.stages, *second.mutation_schedule.stages)} == {.5}
    with pytest.raises(ValueError):
        search_policy('unknown')


def test_search_settings_leave_input_unchanged_and_select_the_safe_forecast_catalog():
    source = {'candidate_seconds': 120, 'tasks': {}}
    snapshot = json.dumps(source, sort_keys=True)
    result = StagedTrainingStudy('staged', 'ts_forecasting').search_settings(source)
    assert json.dumps(source, sort_keys=True) == snapshot
    operations = result['tasks']['ts_forecasting']['operations']
    assert {'forecast_mean', 'forecast_feature_join', 'forecast_window_center'} <= set(operations)
    assert not {'smoothing', 'gaussian_filter'}.intersection(operations)
    assert result['operation_parameters']['fourier_basis']['use_cache'] is False
    assert result['max_arity'] == 4 and result['candidate_seconds'] == 90


def test_parameter_preparation_preserves_selected_alpha(tmp_path):
    from fedot.core.pipelines.adapters import PipelineAdapter
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from benchmark.industrial.experiments.mutation_planning.population_study.preparation import StudyGraphPreparation
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope
    study = StagedTrainingStudy('staged', 'regression')
    with industrial_extension_scope():
        adapter = PipelineAdapter()
        pipeline = PipelineBuilder().add_node('quantile_extractor').add_node('ridge', params={'alpha': .1}).build()
        base = StudyGraphPreparation.from_parameters(adapter, load_training_protocol()[
                                                     'operation_parameters'], 'regression')
        prepared = study.preparation(base, tmp_path)(adapter.adapt(pipeline))
        actual = adapter.restore(prepared.graph)
        assert actual.root_node.parameters['alpha'] == .1
        assert study.configure_selected(actual, {'ridge': {'alpha': 1.}}).root_node.parameters['alpha'] == .1


@pytest.mark.parametrize('task, model', [('classification', 'logit'), ('classification', 'rf'), ('regression', 'ridge')])
def test_new_default_nodes_can_enter_the_declared_parameter_grid(tmp_path, task, model):
    from fedot.core.pipelines.adapters import PipelineAdapter
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from benchmark.industrial.experiments.mutation_planning.population_study.preparation import StudyGraphPreparation
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope
    study = StagedTrainingStudy('staged', task)
    with industrial_extension_scope():
        adapter = PipelineAdapter()
        pipeline = PipelineBuilder().add_node('quantile_extractor').add_node(model).build()
        base = StudyGraphPreparation.from_parameters(adapter, load_training_protocol()['operation_parameters'], task)
        prepared = study.preparation(base, tmp_path)(adapter.adapt(pipeline))
        assert json.loads(prepared.configuration_json)['nodes'][0]['operation'] == model


def test_training_preview_does_not_execute_or_create_output(tmp_path, capsys, monkeypatch):
    from benchmark.industrial.experiments.staged_mutations.__main__ import main
    from benchmark.industrial.experiments.staged_mutations import training_runner
    monkeypatch.setattr(training_runner, 'run_matrix', lambda *args, **kwargs: pytest.fail('Training in preview'))
    target = tmp_path / 'unused'
    monkeypatch.setattr(sys, 'argv', ['staged_mutations', '--train', '--output-dir', str(target)])
    main()
    assert json.loads(capsys.readouterr().out)['execution'] is False
    assert not target.exists()


def test_frozen_sources_reject_a_changed_procedure(tmp_path, monkeypatch):
    from benchmark.industrial.experiments.staged_mutations import training_runner
    source = tmp_path / 'source.py'
    source.write_text('x = 1\n', encoding='utf-8')
    root = tmp_path / 'run'
    root.mkdir()
    monkeypatch.setattr(training_runner, 'implementation_paths', lambda: [source])
    first = training_runner.freeze(root)
    assert training_runner.freeze(root) == first
    source.write_text('x = 2\n', encoding='utf-8')
    with pytest.raises(RuntimeError, match='Frozen source'):
        training_runner.freeze(root)
