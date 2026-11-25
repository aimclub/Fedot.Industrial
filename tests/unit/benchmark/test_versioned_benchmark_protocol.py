from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from benchmark.industrial.protocols import ProtocolError, PreflightStatus, preflight_protocol, resolve_protocol
from benchmark.industrial.protocols.schema import DatasetRelease


EXAMPLE = Path(__file__).resolve().parents[3] / 'benchmark/industrial/examples/protocol_smoke.json'


def read_example():
    return json.loads(EXAMPLE.read_text(encoding='utf-8'))


@pytest.fixture
def payload():
    pytest.importorskip('roast.core.config')
    return read_example()


@pytest.fixture
def forecast(payload):
    payload['suite']['task_kind'] = 'forecasting'
    profile = payload['profiles']['toy']
    profile['release']['unit'] = 'series'
    profile['protocol']['inputs'] = {'target': 'y', 'observed': ['past_temperature'], 'known_future': ['calendar']}
    profile['protocol']['split'] = {
        'kind': 'temporal',
        'frequency': 'H',
        'cases': [
            {'item_id': 'v1', 'series_id': 's1', 'partition': 'validation', 'origin': 10,
             'fit_end': 9, 'target_end': 12, 'horizon': 2},
        ],
    }
    return payload


def protocol(payload):
    return payload['profiles']['toy']['protocol']


def reverse_keys(value):
    if isinstance(value, dict):
        return {key: reverse_keys(item) for key, item in reversed(list(value.items()))}
    if isinstance(value, list):
        return [reverse_keys(item) for item in value]
    return value


def test_resolves_native_roast_contract_and_round_trips(payload):
    from roast.core.config import BenchmarkSuiteConfig

    plan = resolve_protocol(payload)
    assert isinstance(plan.suite, BenchmarkSuiteConfig)
    assert plan.suite.run.random_seed == 42
    assert plan.suite.run.options['evaluation_budget']['max_evaluations'] == 1
    assert plan.suite.models[0].adapter.name == 'majority_class'
    assert resolve_protocol(json.loads(json.dumps(plan.to_dict()))) == plan
    assert resolve_protocol(reverse_keys(payload)) == plan
    payload['suite']['datasets'][0]['provider']['name'] = 'changed'
    assert plan.suite.datasets[0].provider.name == 'local_supervised'
    with pytest.raises(FrozenInstanceError):
        plan.profiles[0].release.source = 'changed'


@pytest.mark.parametrize('path,value', [
    (('release', 'version'), '2'),
    (('release', 'sha256'), '0' * 64),
    (('protocol', 'version'), '2'),
    (('protocol', 'mode'), 'train_only'),
    (('protocol', 'split', 'validation'), ['other']),
    (('protocol', 'metrics', 'accuracy', 'version'), '2'),
    (('protocol', 'metrics', 'accuracy', 'aggregation'), 'mean_by_group'),
])
def test_comparison_key_includes_semantic_changes(payload, path, value):
    original = resolve_protocol(payload)
    node = payload['profiles']['toy']
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    changed = resolve_protocol(payload)
    assert changed.comparison_key != original.comparison_key
    assert changed.plan_id != original.plan_id


def test_execution_options_change_plan_not_comparison_key(payload):
    original = resolve_protocol(payload)
    payload['suite']['models'][0]['adapter']['options'] = {'constant': 'x'}
    payload['suite']['run']['random_seed'] = 9
    payload['suite']['run']['options']['evaluation_budget']['max_evaluations'] = 2
    payload['profiles']['toy']['release']['artifact'] = 'cached/toy.csv'
    changed = resolve_protocol(payload)
    assert changed.comparison_key == original.comparison_key
    assert changed.plan_id != original.plan_id


def test_evaluation_repeats_are_optional_but_identity_bound(payload):
    original = resolve_protocol(payload)
    assert original.evaluation_repeats is None
    options = payload['suite']['run']['options']
    options['evaluation_repeats'] = 2
    updated = resolve_protocol(payload)
    assert updated.evaluation_repeats == 2
    assert updated.plan_id != original.plan_id
    assert updated.comparison_key == original.comparison_key
    assert resolve_protocol(updated.to_dict()) == updated
    for invalid in (None, False, 0, -1, 1.5):
        options['evaluation_repeats'] = invalid
        with pytest.raises(ProtocolError, match='evaluation_repeats'):
            resolve_protocol(payload)


def test_rejects_forged_plan_and_unknown_versions(payload):
    resolved = json.loads(json.dumps(resolve_protocol(payload).to_dict()))
    resolved['suite']['run']['random_seed'] += 1
    with pytest.raises(ProtocolError, match='identities'):
        resolve_protocol(resolved)
    resolved.pop('comparison_key')
    with pytest.raises(ProtocolError, match='require both'):
        resolve_protocol(resolved)
    payload['version'] = 'benchmark_industrial_protocol@2'
    with pytest.raises(ProtocolError, match='Unsupported protocol version'):
        resolve_protocol(payload)


def test_rejects_suite_errors_and_discarded_roast_fields(payload):
    draft = deepcopy(payload)
    draft['suite']['metrics'][0]['schema_version'] = 2
    with pytest.raises(ProtocolError, match='schema_version'):
        resolve_protocol(draft)
    draft = deepcopy(payload)
    draft['suite']['run']['random_sed'] = 2
    with pytest.raises(ProtocolError, match='random_sed'):
        resolve_protocol(draft)
    draft = deepcopy(payload)
    draft['suite']['run']['options']['evaluation_budget']['timeout_seconds'] = True
    with pytest.raises(ProtocolError, match='timeout_seconds'):
        resolve_protocol(draft)
    payload['suite']['run']['primary_metric'] = 'unknown'
    with pytest.raises(ProtocolError, match='primary_metric'):
        resolve_protocol(payload)


def test_rejects_missing_profile_or_metric_rule(payload):
    draft = deepcopy(payload)
    draft['profiles'] = {}
    with pytest.raises(ProtocolError, match='dataset IDs'):
        resolve_protocol(draft)
    protocol(payload)['metrics'] = {}
    with pytest.raises(ProtocolError, match='ROAST metric IDs'):
        resolve_protocol(payload)


@pytest.mark.parametrize('change,reason', [
    ({'validation': ['sample-1']}, 'disjoint'),
    ({'train': ['sample-1', 'sample-1']}, 'duplicates'),
    ({'groups': {}}, 'exactly'),
    ({'groups': {'sample-1': 'a'}}, 'exactly'),
    ({'groups': {'sample-1': 'a', 'sample-2': 'a'}}, 'multiple partitions'),
    ({'test': ['held-out']}, 'forbids test'),
])
def test_rejects_split_leakage(payload, change, reason):
    protocol(payload)['split'].update(change)
    with pytest.raises(ProtocolError, match=reason):
        resolve_protocol(payload)


def test_official_mode_requires_explicit_test_and_distinct_groups(payload):
    spec = protocol(payload)
    spec['mode'] = 'official'
    with pytest.raises(ProtocolError, match='requires a test partition'):
        resolve_protocol(payload)
    spec['split'].update(test=['held-out'], groups={'sample-1': 'a', 'sample-2': 'b', 'held-out': 'c'})
    plan = resolve_protocol(payload)
    assert plan.profiles[0].protocol.split.test == ('held-out',)


@pytest.mark.parametrize('update,reason', [
    ({'fit_end': 11}, 'fit_end <= origin'),
    ({'target_end': 10}, 'origin < target_end'),
    ({'horizon': 3}, 'origin \\+ horizon'),
    ({'origin': '2026-01-01'}, 'same coordinate type'),
    ({'horizon': False}, 'horizon'),
])
def test_rejects_invalid_forecast_boundaries(forecast, update, reason):
    protocol(forecast)['split']['cases'][0].update(update)
    with pytest.raises(ProtocolError, match=reason):
        resolve_protocol(forecast)


def test_rejects_validation_target_overlap_and_future_target(forecast):
    spec = protocol(forecast)
    spec['mode'] = 'official'
    cases = spec['split']['cases']
    cases.append({'item_id': 'test', 'series_id': 's1', 'partition': 'test', 'origin': 11,
                  'fit_end': 10, 'target_end': 13, 'horizon': 2})
    with pytest.raises(ProtocolError, match='must not overlap'):
        resolve_protocol(forecast)
    cases[1].update(origin=12, target_end=14)
    assert len(resolve_protocol(forecast).profiles[0].protocol.split.cases) == 2
    spec['inputs']['known_future'].append('y')
    with pytest.raises(ProtocolError, match='known-future'):
        resolve_protocol(forecast)


def test_forecast_time_offsets_are_canonical(forecast):
    case = protocol(forecast)['split']['cases'][0]
    case.update(origin='2026-01-01T03:00:00+03:00', fit_end='2025-12-31T23:00:00Z',
                target_end='2026-01-01T02:00:00Z')
    original = resolve_protocol(forecast)
    case['origin'] = '2026-01-01T00:00:00Z'
    assert resolve_protocol(forecast) == original
    assert original.profiles[0].protocol.split.cases[0].origin.endswith('Z')
    case['origin'] = '2026-01-01T01:00:00+00:60'
    with pytest.raises(ProtocolError, match='timestamp'):
        resolve_protocol(forecast)


def test_rejects_preprocessing_fitted_on_test(payload):
    protocol(payload)['preprocessing'][0]['fit_on'] = 'test'
    with pytest.raises(ProtocolError, match='fit_on'):
        resolve_protocol(payload)


@pytest.mark.parametrize('artifact', [
    '../toy.csv', 'sub/../toy.csv', '/toy.csv', 'C:/toy.csv', 'a\\toy.csv', 'a/COM1.csv', 'folder /toy.csv',
])
def test_release_rejects_nonportable_paths(artifact):
    release = read_example()['profiles']['toy']['release']
    release['artifact'] = artifact
    with pytest.raises(ProtocolError, match='portable relative'):
        DatasetRelease.parse(release, 'release')


def test_preflight_distinguishes_availability_integrity_and_license(payload, tmp_path):
    plan = resolve_protocol(payload)
    assert preflight_protocol(plan).checks[0].status is PreflightStatus.UNAVAILABLE
    assert preflight_protocol(plan, tmp_path).checks[0].status is PreflightStatus.UNAVAILABLE
    artifact = tmp_path / 'toy.csv'
    artifact.write_bytes(b'abc')
    assert preflight_protocol(plan, tmp_path).ready
    payload['profiles']['toy']['release']['license'] = None
    unknown = resolve_protocol(payload)
    assert [check.status for check in preflight_protocol(unknown, tmp_path).checks] == [PreflightStatus.UNKNOWN_LICENSE]
    artifact.write_bytes(b'bad')
    report = preflight_protocol(unknown, tmp_path)
    assert not report.ready
    assert {check.status for check in report.checks} == {PreflightStatus.UNKNOWN_LICENSE, PreflightStatus.CORRUPT}
    assert report.checks[0].dataset_id == 'toy'


def test_preflight_rejects_symlink_escape(payload, tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    outside = tmp_path / 'outside.csv'
    outside.write_bytes(b'abc')
    try:
        (root / 'toy.csv').symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip('Symlinks unavailable on this host')
    assert preflight_protocol(resolve_protocol(payload), root).checks[0].status is PreflightStatus.INCOMPATIBLE


def test_cli_resolves_and_preflights_without_persisting(payload, tmp_path, capsys):
    from benchmark.industrial.protocols.__main__ import main

    assert main([str(EXAMPLE)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['plan_id'] == resolve_protocol(payload).plan_id
    assert main([str(EXAMPLE), '--preflight', '--data-root', str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out)['checks'][0]['status'] == 'unavailable'
    (tmp_path / 'toy.csv').write_bytes(b'abc')
    assert main([str(EXAMPLE), '--preflight', '--data-root', str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)['ready']
    assert [path.name for path in tmp_path.iterdir()] == ['toy.csv']


def test_cli_rejects_duplicate_json_keys_without_resolving(tmp_path, capsys):
    from benchmark.industrial.protocols.__main__ import main

    manifest = tmp_path / 'duplicate.json'
    manifest.write_text('{"version":"first","version":"second"}', encoding='utf-8')
    assert main([str(manifest)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'incompatible'
    assert 'duplicate object keys' in result['detail']


def test_cli_reports_excessively_nested_json_as_incompatible(tmp_path):
    manifest = tmp_path / 'nested.json'
    manifest.write_text('[' * 2000 + '0' + ']' * 2000, encoding='utf-8')
    command = (
        'import sys; from benchmark.industrial.protocols.__main__ import main; '
        'sys.setrecursionlimit(256); raise SystemExit(main(sys.argv[1:]))'
    )
    completed = subprocess.run([sys.executable, '-c', command, str(manifest)], cwd=EXAMPLE.parents[3],
                               check=False, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 2, completed.stderr
    result = json.loads(completed.stdout)
    assert result['status'] == 'incompatible' and 'nesting depth' in result['detail']


def test_roast_is_optional_for_legacy_and_error_is_actionable():
    command = """
from benchmark.industrial.core import DatasetSpec
from benchmark.industrial.protocols import PROTOCOL_VERSION, MissingRoastError, resolve_protocol
assert DatasetSpec('ucr', 'toy').dataset_name == 'toy'
try:
    resolve_protocol({'version': PROTOCOL_VERSION, 'suite': {}, 'profiles': {}})
except MissingRoastError as error:
    assert 'benchmark-protocol extra' in str(error)
else:
    raise AssertionError('optional ROAST was silently ignored')
"""
    completed = subprocess.run([sys.executable, '-S', '-c', command], cwd=EXAMPLE.parents[3],
                               env={**os.environ, 'PYTHONPATH': ''}, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
