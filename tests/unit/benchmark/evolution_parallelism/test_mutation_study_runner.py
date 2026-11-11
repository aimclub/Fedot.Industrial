import json
import subprocess

import pytest

from benchmark.industrial.experiments.mutation_diversity import runner
from benchmark.industrial.experiments.evolution_parallelism import supervisor


@pytest.fixture
def matrix_input(tmp_path, monkeypatch):
    protocol = dict(training_tasks=['classification'], seeds=[42, 123],
                    arms={'baseline': 'params_mutation_strategy', 'structural': 'structural_balanced'},
                    budget_seconds=300)
    monkeypatch.setattr(runner, 'load_protocol', lambda: protocol)
    source = tmp_path / 'owned_frozen_source.py'
    source.write_text('pass\n', encoding='utf-8')
    monkeypatch.setattr(runner.MutationStudy, 'implementation_paths', lambda self: [source])
    generation = tmp_path / 'generation'
    generation.mkdir()
    value = dict(training_calls=0, initial_configuration_ids={'classification': ['initial']}, rows=[
        dict(task='classification', arm='baseline', configuration_ids=['known']),
        dict(task='classification', arm='structural', configuration_ids=['known', 'novel'])])
    (generation / 'generation.json').write_text(json.dumps(value), encoding='utf-8')
    return generation, value, tmp_path / 'matrix', source


def test_matrix_preserves_fixed_order_and_infrastructure_failures(matrix_input, monkeypatch):
    generation, _, output, _ = matrix_input
    calls = []

    def execute(command, **limits):
        calls.append((command, limits))
        if len(calls) == 2:
            return 1
        if len(calls) == 3:
            raise subprocess.TimeoutExpired(command, 1200)
        return 0

    monkeypatch.setattr(supervisor, 'run_guarded', execute)
    jobs = runner.run_matrix(output, generation)
    assert [(r['seed'], r['arm']) for r in jobs] == [
        (42, 'baseline'), (42, 'structural'), (123, 'structural'), (123, 'baseline')]
    assert [r['status'] for r in jobs] == ['finished', 'failed', 'timed_out', 'finished']
    assert json.loads((output / 'jobs.json').read_text('utf-8')) == jobs
    for command, limits in calls:
        assert '--worker' in command and '--execute' in command
        assert limits == dict(native_threads=1, search_seconds=300)
    with pytest.raises(FileExistsError):
        runner.run_matrix(output, generation)


@pytest.mark.parametrize('training_calls,structural_ids', [(1, ['novel']), (0, ['known', 'initial'])])
def test_matrix_blocks_training_if_proposal_gate_has_no_valid_evidence(
        matrix_input, monkeypatch, training_calls, structural_ids):
    generation, value, output, _ = matrix_input
    value['training_calls'] = training_calls
    value['rows'][1]['configuration_ids'] = structural_ids
    (generation / 'generation.json').write_text(json.dumps(value), encoding='utf-8')
    calls = []
    monkeypatch.setattr(supervisor, 'run_guarded', lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError):
        runner.run_matrix(output, generation)
    assert calls == []


def test_matrix_refuses_mixed_source_versions(matrix_input, monkeypatch):
    generation, _, output, owned_source = matrix_input
    calls = []

    def execute(command, **limits):
        calls.append(command)
        owned_source.write_text('changed = True\n', encoding='utf-8')
        return 0

    monkeypatch.setattr(supervisor, 'run_guarded', execute)
    with pytest.raises(RuntimeError, match='Frozen implementation changed'):
        runner.run_matrix(output, generation)
    assert len(calls) == 1
