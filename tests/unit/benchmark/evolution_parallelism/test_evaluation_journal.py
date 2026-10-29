import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given, settings, strategies as st

from benchmark.industrial.experiments.evolution_parallelism.journal import EvaluationJournal, read_evaluation_journal


def record(number, configuration='same'):
    return dict(evaluation_id=str(number), fitness=[float(number)],
                evaluation_context=dict(configuration_id=configuration))


def test_append_counts_idempotence_and_legacy_snapshot(tmp_path):
    journal = EvaluationJournal(tmp_path)
    journal.append([record(1), record(2)])
    journal.append([record(2), record(3, 'other')])
    assert journal.count == 3 and journal.unique_configurations == 2
    assert not (tmp_path / 'completed_evaluations.json').exists()
    expected = [record(1), record(2), record(3, 'other')]
    assert list(journal.finalize()) == expected
    assert json.loads((tmp_path / 'completed_evaluations.json').read_text()) == expected
    with pytest.raises(FileExistsError):
        EvaluationJournal(tmp_path)


def test_conflicting_batch_does_not_partially_append(tmp_path):
    journal = EvaluationJournal(tmp_path)
    journal.append([record(1)])
    before = journal.path.read_bytes()
    with pytest.raises(ValueError, match='Conflicting'):
        journal.append([record(2), record(1, 'changed')])
    assert journal.path.read_bytes() == before and journal.count == 1
    with pytest.raises(ValueError):
        journal.append([dict(evaluation_id='nonfinite', fitness=[float('nan')])])


def test_interrupted_tail_is_explicit_but_corrupt_complete_lines_fail(tmp_path):
    path = tmp_path / 'records.jsonl'
    path.write_bytes(json.dumps(record(1)).encode() + b'\n{"evaluation_id":')
    result = read_evaluation_journal(path)
    assert result.records == (record(1),) and result.incomplete_tail_bytes > 0
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='line 2'):
        read_evaluation_journal(path)


def test_read_rejects_conflicting_duplicates_and_nonfinite_records(tmp_path):
    path = tmp_path / 'records.jsonl'
    path.write_text(json.dumps(record(1)) + '\n' + json.dumps(record(1, 'changed')) + '\n')
    with pytest.raises(ValueError, match='Conflicting'):
        read_evaluation_journal(path)
    path.write_text('{"evaluation_id":"a","fitness":[NaN]}\n')
    with pytest.raises(ValueError, match='Invalid complete'):
        read_evaluation_journal(path)


def test_unconfirmed_io_never_updates_counts_or_claims_success(tmp_path, monkeypatch):
    from benchmark.industrial.experiments.evolution_parallelism import journal as module
    journal = EvaluationJournal(tmp_path)

    def failed(*args):
        raise OSError('disk failure')

    monkeypatch.setattr(module.os, 'fsync', failed)
    with pytest.raises(OSError):
        journal.append([record(1)])
    assert journal.count == 0
    with pytest.raises(RuntimeError):
        journal.finalize()


def test_final_snapshot_rejects_modified_confirmed_evidence(tmp_path):
    journal = EvaluationJournal(tmp_path)
    journal.append([record(1)])
    journal.path.write_text(json.dumps(record(1, 'changed')) + '\n')
    with pytest.raises(ValueError, match='does not match'):
        journal.finalize()
    assert not (tmp_path / 'completed_evaluations.json').exists()


@settings(max_examples=25, deadline=None)
@given(st.lists(st.lists(st.integers(min_value=0, max_value=12), max_size=8), max_size=5))
def test_batch_partitioning_and_repetition_preserve_first_seen_records(batches):
    with TemporaryDirectory(prefix='industrial-journal-test-') as directory:
        journal = EvaluationJournal(Path(directory))
        reference = {}
        for batch in batches:
            records = [record(number, str(number % 3)) for number in batch]
            journal.append(records)
            for item in records:
                reference.setdefault(item['evaluation_id'], item)
            assert journal.count == len(reference)
            assert journal.unique_configurations == len({r['evaluation_context']['configuration_id']
                                                         for r in reference.values()})
        assert list(journal.finalize()) == list(reference.values())
