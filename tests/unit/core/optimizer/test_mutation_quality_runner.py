import json
from pathlib import Path

import pytest
import torch

from benchmark.industrial.experiments.mutation_planning.quality_runner import (
    _data, load_quality_trial_config, main, run_quality_trial,
)


def test_preview_does_not_create_artifacts_or_require_fedot_patch(tmp_path, capsys):
    output = tmp_path / 'untouched'
    result = run_quality_trial(output)
    assert result['status'] == 'planned'
    assert not output.exists()
    assert main(['--output-dir', str(output)]) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'planned'


def test_typed_trial_config_round_trip_and_isolated_parameters(tmp_path):
    config = load_quality_trial_config()
    record = config.to_record()
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(record), encoding='utf-8')
    assert load_quality_trial_config(path) == config
    record['models']['classification']['parameters']['model'] = 'changed'
    assert config.to_record()['models']['classification']['parameters']['model'] == 'dt'


@pytest.mark.parametrize('field,value', [
    ('schema_version', 'unknown'), ('max_candidates', True), ('evaluations_per_mode', 0),
    ('sample_count', 10), ('series_length', 2), ('seeds', [42, 42]), ('seeds', [-1]),
    ('tasks', ['ts_forecasting']), ('tasks', []), ('generators', ['same', 'same']),
    ('models', {}), ('unexpected', 5),
])
def test_invalid_trial_configuration_is_rejected(tmp_path, field, value):
    record = load_quality_trial_config().to_record()
    record[field] = value
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(ValueError):
        load_quality_trial_config(path)


@pytest.mark.parametrize('task', ['classification', 'regression'])
def test_training_validation_and_test_rows_are_disjoint_and_replayable(task):
    config = load_quality_trial_config().to_record()
    first = _data(task, config, 42)
    second = _data(task, config, 42)
    train, validation, _, test, _, provenance = first
    assert len(set(train.idx) | set(validation.idx) | set(test.idx)) == config['sample_count']
    assert not set(train.idx) & set(validation.idx)
    assert not set(train.idx) & set(test.idx)
    assert not set(validation.idx) & set(test.idx)
    assert validation.target is None and test.target is None
    assert train.target.shape[0] == len(train.idx)
    assert provenance == second[-1]
    torch.testing.assert_close(train.features, second[0].features)


def test_execution_requires_an_explicit_audited_worktree_before_output(tmp_path):
    with pytest.raises(ValueError, match='compatibility_worktree'):
        run_quality_trial(tmp_path / 'missing', execute=True)
    assert not (tmp_path / 'missing').exists()


@pytest.mark.parametrize('task', ['classification', 'regression'])
@pytest.mark.parametrize('mode', ['legacy', 'planned'])
def test_short_real_trial_reports_heldout_quality_and_costs(task, mode):
    from benchmark.industrial.experiments.mutation_planning.quality_runner import _trial
    from benchmark.industrial.experiments.mutation_planning.quality_cost import validate_run_record
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope

    config = load_quality_trial_config().to_record()
    config.update(sample_count=20, series_length=16, evaluations_per_mode=2)
    with industrial_extension_scope():
        record = _trial(task, 42, mode, config, {'purpose': 'integration_test'})
    assert validate_run_record(record) == record
    assert record['status'] == 'succeeded'
    assert record['source']['selected_on'] == 'validation_only'
    assert record['train_seconds'] >= 0 and record['predict_seconds'] >= 0
    assert record['peak_memory_bytes'] >= 0 and record['memory_scope'] == 'python_tracemalloc'
    assert len(record['source']['successful_candidates']) + len(record['source']['failed_candidates']) == 2


def test_execution_saves_paired_reports_and_never_overwrites_evidence(tmp_path, monkeypatch):
    import fedot
    from benchmark.industrial.experiments.mutation_planning import quality_runner
    from fedot_ind.integration.fedot import compat

    audit = {'worktree': str(Path(fedot.__file__).resolve().parents[1]), 'base_revision': 'test'}
    monkeypatch.setattr(compat, 'audit_compatibility_worktree', lambda _: audit)

    def measured(task, seed, mode, config, source):
        assert source['fedot_compatibility'] == audit
        assert all(isinstance(version, str) for version in source['environment'].values())
        assert source['implementation_sha256']
        assert all(len(digest) == 64 for digest in source['implementation_sha256'].values())
        return dict(task=task, scenario='fixture', dataset=task, seed=seed, mode=mode,
                    metric='test', greater_is_better=True, quality=.5, train_seconds=1., predict_seconds=.1,
                    memory_scope=None, peak_memory_bytes=None, status='succeeded', budget_id='fixture')

    monkeypatch.setattr(quality_runner, '_trial', measured)
    output = tmp_path / 'reports'
    report = run_quality_trial(output, execute=True, compatibility_worktree='fixture')
    assert report['pair_count'] == 4 and report['acceptance']['status'] == 'not_assessed'
    assert json.loads((output / 'quality_cost.json').read_text()) == report
    assert len(json.loads((output / 'records.json').read_text())['records']) == 8
    assert (output / 'quality_cost.md').is_file()
    with pytest.raises(ValueError, match='fresh'):
        run_quality_trial(output, execute=True, compatibility_worktree='fixture')
