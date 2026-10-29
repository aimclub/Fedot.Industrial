"""Coordinator-owned append-only evidence and an atomic legacy JSON snapshot."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile


def _canonical(record):
    if type(record) is not dict or type(record.get('evaluation_id')) is not str or not record['evaluation_id']:
        raise ValueError('An evaluation record needs a nonempty evaluation_id')
    context = record.get('evaluation_context')
    if context is not None and type(context) is not dict:
        raise ValueError('evaluation_context must be an object or absent')
    if context is not None and context.get('configuration_id') is not None:
        if type(context['configuration_id']) is not str:
            raise ValueError('configuration_id must be a string or absent')
    return json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)


@dataclass(frozen=True)
class JournalRead:
    records: tuple[dict, ...]
    incomplete_tail_bytes: int


def read_evaluation_journal(path):
    records, seen = [], {}
    tail = 0
    with Path(path).open('rb') as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith(b'\n'):
                tail = len(line)
                break
            try:
                record = json.loads(line)
                canonical = _canonical(record)
            except (ValueError, UnicodeError) as error:
                raise ValueError(f'Invalid complete evaluation record at line {number}') from error
            identity = record['evaluation_id']
            if identity in seen:
                if seen[identity] != canonical:
                    raise ValueError(f'Conflicting evaluation_id at line {number}')
            else:
                seen[identity] = canonical
                records.append(record)
    return JournalRead(tuple(records), tail)


class EvaluationJournal:
    """One coordinator owns this writer; it never selects or reuses fitness.

    New batches are validated before writing and synced before updating counts.
    A failed write poisons the writer instead of guessing which bytes survived.
    Reading an incomplete tail is explicit; it never resumes an optimiser.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / 'completed_evaluations.jsonl'
        with self.path.open('x', encoding='utf-8'):
            pass
        self._seen, self._configurations = {}, set()
        self._failed = False

    @property
    def count(self):
        return len(self._seen)

    @property
    def unique_configurations(self):
        return len(self._configurations)

    def append(self, records):
        if self._failed:
            raise RuntimeError('Cannot continue an unconfirmed journal write')
        pending = {}
        configurations = set()
        for record in records:
            canonical = _canonical(record)
            identity = record['evaluation_id']
            previous = pending.get(identity, self._seen.get(identity))
            if previous is not None and previous != canonical:
                raise ValueError('Conflicting completed evaluation evidence')
            if previous is None:
                pending[identity] = canonical
                context = record.get('evaluation_context') or {}
                configuration = context.get('configuration_id')
                if configuration is not None:
                    if type(configuration) is not str:
                        raise ValueError('configuration_id must be a string or absent')
                    configurations.add(configuration)
        if not pending:
            return
        try:
            with self.path.open('a', encoding='utf-8', newline='\n') as stream:
                stream.write(''.join(value + '\n' for value in pending.values()))
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            self._failed = True
            raise
        self._seen.update(pending)
        self._configurations.update(configurations)

    def finalize(self):
        if self._failed:
            raise RuntimeError('Cannot finalize an unconfirmed journal write')
        result = read_evaluation_journal(self.path)
        if (result.incomplete_tail_bytes or len(result.records) != self.count
                or any(self._seen.get(row['evaluation_id']) != _canonical(row) for row in result.records)):
            raise ValueError('Journal does not match the confirmed completed evaluations')
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.root,
                                             prefix='completed_', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(result.records, stream, indent=2, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.root / 'completed_evaluations.json')
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return result.records
