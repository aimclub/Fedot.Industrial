"""A hard wall-clock guard for the complete example, including final refit."""
import os
import subprocess
from time import time


def run_guarded(command, *, native_threads=1, timeout_seconds=1200, search_seconds=None):
    import psutil

    environment = dict(os.environ)
    # Apply equally before imports in each isolated example, not to the caller.
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[name] = str(native_threads)
    started = time()
    environment['INDUSTRIAL_SEARCH_START_UNIX'] = str(started)
    if search_seconds is not None:
        environment['INDUSTRIAL_SEARCH_DEADLINE_UNIX'] = str(started + search_seconds)
    process = subprocess.Popen(command, env=environment)
    owned = psutil.Process(process.pid)
    birth = owned.create_time()
    try:
        return process.wait(timeout=timeout_seconds)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        # Never enumerate another task's processes or terminate a reused PID.
        if owned.is_running() and owned.create_time() == birth:
            descendants = owned.children(recursive=True)
            for child in reversed(descendants):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            try:
                owned.kill()
            except psutil.NoSuchProcess:
                pass
            _, alive = psutil.wait_procs(descendants + [owned], timeout=10)
            if alive:
                raise RuntimeError(f'Cannot confirm shutdown of owned processes: {[p.pid for p in alive]}')
        process.wait(timeout=10)
        raise
