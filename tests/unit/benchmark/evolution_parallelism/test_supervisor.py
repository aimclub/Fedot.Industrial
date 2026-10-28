import subprocess
import sys
import time

import psutil
import pytest

from benchmark.industrial.experiments.evolution_parallelism.supervisor import run_guarded


def test_guard_stops_only_its_owned_process_tree(tmp_path):
    pid_path = tmp_path / 'child.pid'
    code = (
        'import subprocess,sys,time,pathlib; '
        'child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]); '
        f'pathlib.Path({str(pid_path)!r}).write_text(str(child.pid)); time.sleep(60)'
    )
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            run_guarded([sys.executable, '-c', code], timeout_seconds=3)
        assert unrelated.poll() is None
        assert pid_path.is_file()
        pid = int(pid_path.read_text())
        for _ in range(30):
            if not psutil.pid_exists(pid):
                break
            time.sleep(.1)
        assert not psutil.pid_exists(pid)
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=10)
