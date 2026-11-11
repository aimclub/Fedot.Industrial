"""Explicit current-thread profiling, without patching algorithms or RNGs."""

import cProfile
import csv
import json
from pathlib import Path
import pstats
import sys
from time import perf_counter, process_time, thread_time
import warnings


def profile_rows(stats):
    """Exclusive times are additive; cumulative times of nested calls are not."""
    rows = []
    for (filename, line, function), (primitive, calls, own, cumulative, _) in stats.items():
        path = filename.replace('\\', '/')
        if 'statistical/' in path:
            category = 'statistics'
        elif 'mutation' in path or 'graph_validation' in path or 'search_policy' in path:
            category = 'planning_validation'
        elif any(part in path for part in ('pickle', 'json/', 'serialization', 'protocol/')):
            category = 'serialization'
        else:
            category = 'other'
        rows.append(dict(file=path, line=line, function=function, primitive_calls=primitive, calls=calls,
                         exclusive_seconds=own, cumulative_seconds=cumulative, category=category))
    return sorted(rows, key=lambda row: (-row['exclusive_seconds'], row['file'], row['line'], row['function']))


class PythonProfile:
    """A diagnostic run, never a clean speed measurement or a worker profile.

    Own only the current thread's profiler. Background monitoring and Dask
    workers require separate scopes. Output rendering is outside body timing.
    """

    def __init__(self, output_dir, *, label):
        self.output_dir, self.label = Path(output_dir), str(label)
        self._active = False

    def __enter__(self):
        if self._active or sys.getprofile() is not None:
            raise RuntimeError('The current thread already owns a profiler')
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.profiler = cProfile.Profile()
        self.started = perf_counter(), process_time(), thread_time()
        self._active = True
        self.profiler.enable()
        return self

    def __exit__(self, error_type, error, traceback):
        self.profiler.disable()
        self._active = False
        wall, process, thread = (clock() - start for clock, start in
                                 zip((perf_counter, process_time, thread_time), self.started))
        try:
            self.profiler.dump_stats(str(self.output_dir / 'calls.pstats'))
            rows = profile_rows(pstats.Stats(self.profiler).stats)
            with (self.output_dir / 'calls.csv').open('w', newline='', encoding='utf-8') as stream:
                fields = tuple(rows[0]) if rows else ('file', 'line', 'function')
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            metadata = dict(schema_version='industrial_python_profile@1', label=self.label,
                            wall_seconds=wall, process_cpu_seconds=process, thread_cpu_seconds=thread,
                            current_thread_only=True, diagnostic_run=True,
                            outcome='succeeded' if error_type is None else 'failed',
                            error_type=None if error_type is None else error_type.__name__,
                            cumulative_times_are_not_additive=True)
            (self.output_dir / 'profile.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        except Exception as export_error:
            if error_type is None:
                raise
            warnings.warn(f'Profile export failed: {export_error}', RuntimeWarning)
        return False


def profile_call(operation, output_dir, *args, label='call', **kwargs):
    with PythonProfile(output_dir, label=label):
        return operation(*args, **kwargs)
