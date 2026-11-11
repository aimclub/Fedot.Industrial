"""Opt-in process-local tracing of FEDOT operations; no patch survives its scope."""
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
from time import perf_counter, time


def exception_chain(error):
    records, seen = [], set()
    while error is not None and id(error) not in seen and len(records) < 16:
        seen.add(id(error))
        records.append({'type': type(error).__name__, 'message': str(error)})
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    return records


@contextmanager
def operation_trace(output_dir, evaluation):
    from fedot.core.operations.operation import Operation

    stack, records, originals = [], [], {}

    def instrument(name, function):
        @wraps(function)
        def measured(self, *args, **kwargs):
            start = perf_counter()
            frame = {'child_seconds': 0.}
            stack.append(frame)
            failure = None
            try:
                return function(self, *args, **kwargs)
            except Exception as error:
                failure = exception_chain(error)
                raise
            finally:
                elapsed = perf_counter() - start
                stack.pop()
                if stack:
                    stack[-1]['child_seconds'] += elapsed
                records.append({**evaluation, 'pid': os.getpid(), 'operation': self.operation_type,
                                'stage': name, 'start_epoch': time() - elapsed, 'inclusive_seconds': elapsed,
                                'exclusive_seconds': max(0., elapsed - frame['child_seconds']),
                                'status': 'failed' if failure else 'succeeded', 'error_chain': failure})
        return measured

    try:
        for name in ('fit', 'predict', 'predict_for_fit'):
            originals[name] = getattr(Operation, name)
            setattr(Operation, name, instrument(name, originals[name]))
        yield
    finally:
        for name, function in originals.items():
            setattr(Operation, name, function)
        if records:
            root = Path(output_dir)
            root.mkdir(parents=True, exist_ok=True)
            with (root / f'operations_{os.getpid()}.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(''.join(json.dumps(row, allow_nan=False) + '\n' for row in records))
