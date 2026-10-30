from hashlib import sha256
from types import SimpleNamespace
from zipfile import ZipFile

from hypothesis import given, strategies as st
import numpy as np
import pytest

from benchmark.industrial.experiments.mutation_quality.freeze import (
    array_record, save_data, seal, verify_freeze, write_json,
)
from benchmark.industrial.experiments.mutation_quality.sources import (
    deliver, extract_electricity_train, train_row, verify_train_provenance,
)


@given(st.text(alphabet='abcXYZ!?0123456789.-', min_size=1, max_size=40))
def test_future_tokens_cannot_affect_training_values(future):
    name, values = train_row(f'T1:2012-01-01 00-00-00:1,2,3,{future},{future}', 2, 5)
    assert name == 'T1' and values == ['1', '2', '3']


@pytest.mark.parametrize('line,tail,length', [
    ('T1:date:1,2,3,4', True, 4), ('T1:date:1,2,3', 2, 5),
    ('T1:date:1,nan,3,4,5', 2, 5), ('T1:date:1,inf,3,4,5', 2, 5),
])
def test_invalid_train_or_boundary_is_rejected(line, tail, length):
    with pytest.raises(ValueError):
        train_row(line, tail, length)


def test_electricity_extraction_does_not_parse_future_or_extract_other_files(tmp_path):
    path = tmp_path / 'source.zip'
    with ZipFile(path, 'w') as zipped:
        zipped.writestr('electricity.tsf', '@attribute series_name string\n@attribute start_timestamp date\n'
                        '@frequency hourly\n@data\nT1:2012-01-01 00-00-00:1,2,3,UNREAD,UNREAD\n')
    spec = dict(member='electricity.tsf', expected_length=5, reserved_tail=2, expected_series=1,
                train_file='electricity-train.csv', page='https://example.invalid/data',
                boundary_reference='https://example.invalid/protocol', sha256=sha256(path.read_bytes()).hexdigest())
    result = extract_electricity_train(path, tmp_path, spec)
    assert result['future_tokens_parsed'] is False
    content = (tmp_path / spec['train_file']).read_text('utf-8')
    assert 'UNREAD' not in content
    assert verify_train_provenance(tmp_path / spec['train_file'], spec) == result
    with pytest.raises(FileExistsError):
        extract_electricity_train(path, tmp_path, spec)
    (tmp_path / spec['train_file']).write_text(content + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='provenance'):
        verify_train_provenance(tmp_path / spec['train_file'], spec)


def test_delivery_validates_existing_file_without_network_or_overwrite(tmp_path, monkeypatch):
    path = tmp_path / 'Hourly-train.csv'
    path.write_bytes(b'original')
    spec = dict(file=path.name, sha256=sha256(b'original').hexdigest(), url='https://example.invalid', maximum_bytes=30)
    monkeypatch.setattr('benchmark.industrial.experiments.mutation_quality.sources.urlopen',
                        lambda *a, **kw: pytest.fail('Unexpected network'))
    assert deliver(spec, tmp_path) == path
    spec['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='checksum'):
        deliver(spec, tmp_path)
    assert path.read_bytes() == b'original'


def make_frozen_fixture(tmp_path):
    original = tmp_path / 'procedure.py'
    original.write_bytes(b'x = 1\n')
    root = tmp_path / 'frozen'
    (root / 'inputs').mkdir(parents=True)
    container = SimpleNamespace(features=np.arange(12, dtype=np.float32).reshape(4, 1, 3), target=np.arange(4),
                                idx=np.arange(4), state='fit', data_type='multi_ts',
                                task=SimpleNamespace(task_type='regression'))
    data = SimpleNamespace(folds=((container, container),), refit=container, test=container,
                           source=dict(official_test_used=False))
    metadata = save_data(root / 'inputs/sample.npz', data)
    with ZipFile(root / 'sources.zip', 'w') as archive:
        archive.writestr('sources/0.py', original.read_bytes())
    write_json(root / 'lock.json', dict(datasets={'sample': metadata}, jobs=[['sample', 'baseline', 42]],
                                        sources={str(original): dict(member='sources/0.py', sha256=sha256(original.read_bytes()).hexdigest())}))
    seal(root)
    return root, original


def test_frozen_arrays_and_embedded_sources_roundtrip(tmp_path):
    root, source = make_frozen_fixture(tmp_path)
    identity = sha256((root / 'seal.json').read_bytes()).hexdigest()
    assert verify_freeze(root, check_current=True, expected_seal_sha256=identity)['verified']
    source.write_text('changed\n', encoding='utf-8')
    assert verify_freeze(root)['verified']  # The archived procedure remains intact.
    with pytest.raises(ValueError, match='Current source'):
        verify_freeze(root, check_current=True)
    with pytest.raises(ValueError, match='trusted anchor'):
        verify_freeze(root, expected_seal_sha256='0' * 64)


@pytest.mark.parametrize('mutation', ['array', 'extra', 'missing', 'nested_seal'])
def test_tampering_and_membership_changes_are_rejected(tmp_path, mutation):
    root, _ = make_frozen_fixture(tmp_path)
    if mutation == 'array':
        (root / 'inputs/sample.npz').write_bytes(b'not the frozen data')
    elif mutation == 'extra':
        (root / 'extra').write_bytes(b'extra')
    elif mutation == 'nested_seal':
        (root / 'inputs/seal.json').write_text('{}', encoding='utf-8')
    else:
        (root / 'inputs/sample.npz').unlink()
    with pytest.raises(ValueError, match='Frozen file'):
        verify_freeze(root)


@pytest.mark.parametrize('value', [np.array([float('nan')]), np.array(['object'], dtype=object)])
def test_non_numeric_or_nonfinite_frozen_arrays_are_rejected(value):
    with pytest.raises(ValueError):
        array_record(value)
