"""Opt-in process-local tracing of FEDOT operations; no patch survives its scope."""
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
from time import perf_counter, time


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
            try:
                return function(self, *args, **kwargs)
            finally:
                elapsed = perf_counter() - start
                stack.pop()
                if stack:
                    stack[-1]['child_seconds'] += elapsed
                records.append({**evaluation, 'pid': os.getpid(), 'operation': self.operation_type,
                                'stage': name, 'start_epoch': time() - elapsed, 'inclusive_seconds': elapsed,
                                'exclusive_seconds': max(0., elapsed - frame['child_seconds'])})
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
