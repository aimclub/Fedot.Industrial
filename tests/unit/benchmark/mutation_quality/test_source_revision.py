import json

import pytest

from benchmark.industrial.experiments.mutation_quality.execution import revision
from benchmark.industrial.experiments.mutation_quality.preflight import file_digest
from tests.unit.benchmark.mutation_quality.test_freeze_sources import make_frozen_fixture


def test_revision_preserves_every_data_byte_and_the_old_bundle(tmp_path, monkeypatch):
    previous, source = make_frozen_fixture(tmp_path)
    identity = file_digest(previous / 'seal.json')
    before = {str(path): file_digest(path) for path in previous.rglob('*') if path.is_file()}
    source.write_text('x = 2\n', encoding='utf-8')
    changes = {str(source): file_digest(source)}
    monkeypatch.setattr(revision, 'freeze_sources', lambda: changes)
    target = tmp_path / 'revision'
    assert revision.revise_sources(previous, target, expected_seal=identity, changes=changes)['verified']
    assert before == {str(path): file_digest(path) for path in previous.rglob('*') if path.is_file()}
    assert file_digest(previous / 'inputs/sample.npz') == file_digest(target / 'inputs/sample.npz')
    old = json.loads((previous / 'lock.json').read_text('utf-8'))
    new = json.loads((target / 'lock.json').read_text('utf-8'))
    assert new['datasets'] == old['datasets'] and new['jobs'] == old['jobs']
    assert new['source_revision']['parent_seal'] == identity


def test_unreviewed_change_is_rejected_before_creating_output(tmp_path, monkeypatch):
    previous, source = make_frozen_fixture(tmp_path)
    source.write_text('unreviewed\n', encoding='utf-8')
    monkeypatch.setattr(revision, 'freeze_sources', lambda: {str(source): file_digest(source)})
    target = tmp_path / 'revision'
    with pytest.raises(ValueError, match='exact reviewed'):
        revision.revise_sources(previous, target, expected_seal=file_digest(previous / 'seal.json'), changes={})
    assert not target.exists()
