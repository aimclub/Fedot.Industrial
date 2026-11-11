from types import SimpleNamespace

import pytest

from benchmark.industrial.experiments.mutation_planning.population_study.forecast_recheck import (
    recheck, require_cpu_build, select_jobs,
)
from benchmark.industrial.experiments.mutation_planning.population_study.protocol import load_protocol
from benchmark.industrial.experiments.mutation_planning.population_study.protocol import experiment_matrix


def manifest():
    protocol = load_protocol()
    return {'protocol': protocol.to_record(), 'protocol_sha256': protocol.digest,
            'matrix': experiment_matrix(protocol)}


def test_diagnostic_selection_preserves_original_jobs_and_order():
    original = manifest()
    protocol, jobs = select_jobs(original, [52, 40, 36])
    assert protocol.to_record() == original['protocol']
    assert [job['ordinal'] for job in jobs] == [36, 40, 52]
    assert all(job['mode'] == 'legacy' for job in jobs)
    assert all(job['protocol_sha256'] == protocol.digest for job in jobs)


@pytest.mark.parametrize('ordinals', [[], [True], [-1], [36, 36], [54], [0], [36, 0]])
def test_nonforecast_or_ambiguous_jobs_are_rejected(ordinals):
    with pytest.raises(ValueError):
        select_jobs(manifest(), ordinals)


def test_diagnostics_do_not_accept_a_changed_protocol_digest():
    original = manifest()
    original['protocol']['search_seconds'] += 1
    with pytest.raises(ValueError, match='digest'):
        select_jobs(original, [36])


def test_diagnostics_reject_a_frozen_matrix_that_disagrees_with_protocol():
    original = manifest()
    original['matrix'] = list(original['matrix'])[:-1]
    with pytest.raises(ValueError, match='matrix'):
        select_jobs(original, [36])


@pytest.mark.parametrize('cuda_build, available', [('12.1', False), ('12.1', True), (None, True)])
def test_cpu_admission_rejects_masked_cuda_builds(cuda_build, available):
    runtime = SimpleNamespace(version=SimpleNamespace(cuda=cuda_build),
                              cuda=SimpleNamespace(is_available=lambda: available))
    with pytest.raises(RuntimeError, match='CPU-only'):
        require_cpu_build(runtime)


def test_real_cpu_build_is_admissible():
    runtime = SimpleNamespace(version=SimpleNamespace(cuda=None),
                              cuda=SimpleNamespace(is_available=lambda: False))
    require_cpu_build(runtime)


def test_snapshot_covers_runtime_adapters_and_portable_fedot_patches():
    from benchmark.industrial.experiments.mutation_planning.population_study.forecast_recheck import (
        _diagnostic_snapshot,
    )
    snapshot = _diagnostic_snapshot()
    for path in ('fedot_ind/integration/fedot/extensions/factories.py',
                 'fedot_ind/integration/fedot/extensions/manifest.py',
                 'fedot_ind/integration/fedot/compat/manifest.json'):
        assert len(snapshot[path]) == 64
    assert any(path.endswith('.patch') for path in snapshot)


def test_dry_run_does_not_create_output_or_import_training(monkeypatch, tmp_path):
    import json
    from benchmark.industrial.experiments.mutation_planning.population_study import forecast_recheck
    source, output = tmp_path / 'original', tmp_path / 'diagnostics'
    source.mkdir()
    original = json.dumps(manifest())
    (source / 'frozen_manifest.json').write_text(original, encoding='utf-8')
    monkeypatch.setattr(forecast_recheck, 'load_study_data', lambda *args: pytest.fail('Training data loaded'))
    result = recheck(source, output, [36, 39, 40, 47, 52])
    assert result['status'] == 'planned'
    assert result['purpose'] == 'diagnostic_only_not_full_acceptance'
    assert not output.exists()
    assert (source / 'frozen_manifest.json').read_text(encoding='utf-8') == original


@pytest.mark.parametrize('damage', [None, 'ordinal', 'seed', 'mode', 'protocol_sha256', 'scenario',
                                    'source', 'implementation', 'compatibility', 'exit', 'native_fault'])
def test_execution_rejects_mismatched_or_failed_evidence_without_training(monkeypatch, tmp_path, damage):
    import json
    import fedot
    from benchmark.industrial.experiments.mutation_planning.population_study import forecast_recheck as module
    from fedot_ind.integration.fedot import compat

    source, output, worktree = tmp_path / 'original', tmp_path / 'diagnostics', tmp_path / 'fedot'
    source.mkdir()
    raw = manifest()
    raw['data_sources'] = {'D1:42': {'fixture': 'same-data'}}
    (source / 'frozen_manifest.json').write_text(json.dumps(raw), encoding='utf-8')
    monkeypatch.setattr(fedot, '__file__', str(worktree / 'fedot/__init__.py'))
    audit_state = {'revision': 'original'}
    monkeypatch.setattr(compat, 'audit_compatibility_worktree',
                        lambda path: {'worktree': str(worktree), **audit_state})
    monkeypatch.setattr(module, 'require_cpu_build', lambda runtime: None)
    monkeypatch.setattr(module, 'load_study_data', lambda *args: SimpleNamespace(source={'fixture': 'same-data'}))
    revision = {'current': 'original'}
    monkeypatch.setattr(module, '_diagnostic_snapshot', lambda: dict(revision))

    def worker(job_path, timeout):
        assert timeout == raw['protocol']['run_seconds']
        payload = json.loads(job_path.read_text(encoding='utf-8'))
        result = {**payload['job'], 'status': 'succeeded', 'source': {'fixture': 'same-data'},
                  'quality_loss': 0.25, 'search_seconds': 1.0, 'selected_graph': 'mssa_forecaster'}
        changes = {'ordinal': 39, 'seed': 123, 'mode': 'planned', 'protocol_sha256': 'different',
                   'scenario': {'name': 'D2', 'task': 'ts_forecasting'}, 'source': {'fixture': 'other-data'}}
        if damage in changes:
            result[damage] = changes[damage]
        if damage == 'implementation':
            revision['current'] = 'modified'
        if damage == 'compatibility':
            audit_state['revision'] = 'modified'
        (job_path.parent / 'result.json').write_text(json.dumps(result), encoding='utf-8')
        (job_path.parent / 'run.log').write_text(
            'Fatal Python error: fixture' if damage == 'native_fault' else '', encoding='utf-8')
        return 1 if damage == 'exit' else 0

    monkeypatch.setattr(module, 'execute_worker', worker)
    if damage in ('ordinal', 'seed', 'mode', 'protocol_sha256', 'scenario', 'source',
                  'implementation', 'compatibility'):
        with pytest.raises(ValueError):
            recheck(source, output, [36], worktree=worktree, execute=True)
        assert not (output / 'records.json').exists()
    else:
        report = recheck(source, output, [36], worktree=worktree, execute=True)
        assert not report['full_study_accepted']
        assert report['results'][0]['status'] == ('succeeded' if damage is None else 'failed')
