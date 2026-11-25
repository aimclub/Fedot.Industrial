from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

from benchmark.industrial.experiments.supervised_pilot.data import (
    digest, load_train, read_development, seal_tsc, tsc_splits,
)
from benchmark.industrial.experiments.supervised_pilot.protocol import parse_tsc_protocol, tsc_protocol
from benchmark.industrial.protocols.assets import AssetCorrupt, AssetUnavailable, LocalAsset
from benchmark.industrial.protocols.schema import ProtocolError


def test_protocol_parses_without_local_sources_and_preserves_order():
    protocol = tsc_protocol()
    payload = protocol.as_payload()
    assert parse_tsc_protocol(payload) == protocol
    assert protocol.tasks()[:2] == (('ECG200', 42), ('ECG200', 123))
    assert len(protocol.tasks()) == 8
    with pytest.raises(ProtocolError, match='unknown'):
        parse_tsc_protocol({**payload, 'test_root': 'unused'})
    with pytest.raises(ProtocolError, match='ordered dataset'):
        parse_tsc_protocol({**payload, 'datasets': list(reversed(payload['datasets']))})


@pytest.fixture
def fixture_protocol(tmp_path):
    base = tsc_protocol()
    profile = replace(base.profile, sklearn=version('scikit-learn'))
    target = np.array(['even', 'odd'] * 20)
    content = ''.join(f'{label}\t{index}\t{-index}\n' for index, label in enumerate(target)).encode()
    asset = LocalAsset('ECG200_TRAIN.tsv', digest(content))
    source = replace(base.datasets[0], asset=asset, shape=(40, 1, 2), development_size=32,
                     split_sha256=tsc_splits(target, profile).identity)
    (tmp_path / asset.path).write_bytes(content)
    return replace(base, profile=profile, datasets=(source,))


def test_sealed_development_does_not_read_or_contain_assessment(tmp_path, fixture_protocol, monkeypatch):
    root = tmp_path / 'frozen'
    seal = seal_tsc(fixture_protocol, tmp_path, root)
    assert seal == seal_tsc(fixture_protocol, tmp_path, tmp_path / 'another')
    with pytest.raises(FileExistsError):
        seal_tsc(fixture_protocol, tmp_path, root)
    opened = []
    real_open = Path.open

    def record_open(path, *args, **kwargs):
        opened.append(path)
        assert 'assessment' not in path.parts and '_TEST' not in path.name
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', record_open)
    dev = root / 'ECG200' / 'development'
    panel, description = read_development(dev, seal['datasets'][0]['development'])
    assert opened == [dev / 'receipt.json', dev / 'panel.npz']
    assert panel.features.shape == (32, 1, 2)
    np.testing.assert_array_equal(panel.features[:, 0, 0], panel.rows)
    assert 'assessment' not in description
    assert tuple(panel.rows) == tuple(description['development'])
    assessment_id = next(i for i in range(40) if i not in panel.rows)
    with pytest.raises(ProtocolError, match='unavailable'):
        panel.take([assessment_id])
    for fold in description['inner']:
        assert not set(fold['fit']) & set(fold['score'])
        assert set(fold['epoch_fit']) | set(fold['epoch_validation']) == set(fold['fit'])


def test_missing_train_cannot_read_test(tmp_path, fixture_protocol, monkeypatch):
    train = tmp_path / 'ECG200_TRAIN.tsv'
    train.rename(tmp_path / 'ECG200_TEST.tsv')
    source = fixture_protocol.datasets[0]

    def forbidden(*args, **kwargs):
        pytest.fail('A missing TRAIN source must not open a file')

    monkeypatch.setattr(Path, 'open', forbidden)
    with pytest.raises(AssetUnavailable):
        load_train(tmp_path, source)
    with pytest.raises(ProtocolError, match='TRAIN'):
        replace(source, asset=LocalAsset('ECG200_TEST.tsv', source.asset.sha256))


def test_a_changed_source_or_split_cannot_create_a_seal(tmp_path, fixture_protocol):
    source = fixture_protocol.datasets[0]
    wrong_split = replace(fixture_protocol, datasets=(replace(source, split_sha256='a' * 64),))
    bundle = tmp_path / 'seal'
    with pytest.raises(ProtocolError, match='split SHA-256'):
        seal_tsc(wrong_split, tmp_path, bundle)
    train = tmp_path / source.asset.path
    train.write_bytes(train.read_bytes() + b'\n')
    with pytest.raises(AssetCorrupt, match='SHA-256'):
        seal_tsc(fixture_protocol, tmp_path, bundle)
    assert not bundle.exists()


def test_development_file_changes_and_wrong_roles_are_rejected(tmp_path, fixture_protocol):
    seal = seal_tsc(fixture_protocol, tmp_path, tmp_path / 'seal')
    links = seal['datasets'][0]
    root = tmp_path / 'seal' / 'ECG200'
    with pytest.raises(ProtocolError, match='development receipt'):
        read_development(root / 'assessment', links['assessment'])
    (root / 'development' / 'panel.npz').write_bytes(b'tampered')
    with pytest.raises(AssetCorrupt):
        read_development(root / 'development', links['development'])
