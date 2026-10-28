import json

import pytest

from benchmark.industrial.experiments.evolution_parallelism.acceptance import validate_pair


def fixture_pair(root):
    for run in ('sequential', 'dask'):
        directory = root / run
        (directory / 'traces').mkdir(parents=True)
        summary = dict(status='succeeded', minimum_generations_met=True,
                       source=dict(train_sha256='train', folds=[1, 2, 3], test_used_for_search=False),
                       torch_flags_before={'threads': 1}, torch_flags_after={'threads': 1},
                       test_loss=1., search_wall_seconds=2., wall_seconds=2., phases={},
                       dask_sessions=1, owned_sessions_closed=True, diagnostics={'evaluations_reused': 44})
        files = {'summary.json': summary, 'selected_graph.json': {'model': 'ridge'},
                 'supervisor.json': {'end_to_end_seconds': 3.},
                 'completed_evaluations.json': [{'evaluation_context': {'configuration_id': 'one'}, 'fitness': [1.]}]}
        for name, value in files.items():
            (directory / name).write_text(json.dumps(value), encoding='utf-8')
        (directory / 'traces/evaluations_1.jsonl').write_text(json.dumps(
            {'configuration_id': 'one', 'valid': True, 'fitness': 1.}) + '\n', encoding='utf-8')


def test_existing_golem_fitness_reuse_is_not_mistaken_for_a_new_cache(tmp_path):
    fixture_pair(tmp_path)
    result = validate_pair(tmp_path)
    assert result['passed'] and result['checks']['same_existing_fitness_reuse']
    assert result['input_sha256'] and result['validator_sha256']
    original = (tmp_path / 'acceptance.json').read_bytes()
    with pytest.raises(FileExistsError):
        validate_pair(tmp_path)
    assert (tmp_path / 'acceptance.json').read_bytes() == original


@pytest.mark.parametrize('field,value', [('test_loss', 2.), ('owned_sessions_closed', False),
                                         ('source', {'train_sha256': 'changed', 'folds': [
                                          1, 2], 'test_used_for_search': True}),
                                         ('torch_flags_after', {'threads': 2}), ('diagnostics', {'evaluations_reused': 0})])
def test_mismatches_fail_acceptance(tmp_path, field, value):
    fixture_pair(tmp_path)
    path = tmp_path / 'dask/summary.json'
    summary = json.loads(path.read_text('utf-8'))
    summary[field] = value
    path.write_text(json.dumps(summary), encoding='utf-8')
    assert not validate_pair(tmp_path)['passed']


def test_changed_actual_attempts_fail_even_with_identical_success_archive(tmp_path):
    fixture_pair(tmp_path)
    path = tmp_path / 'dask/traces/evaluations_1.jsonl'
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'configuration_id': 'extra', 'valid': False, 'fitness': None}) + '\n')
    result = validate_pair(tmp_path)
    assert not result['passed'] and result['checks']['same_evaluated_pool_and_fitness']
