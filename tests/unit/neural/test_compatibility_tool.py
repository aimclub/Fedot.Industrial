"""Unsafe checkpoint reads require an explicit, local trust decision."""
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[3] / "tools/neural_compatibility_check.py"


def test_verify_does_not_load_pickle_or_write_reports_without_trust(tmp_path):
    output, report = tmp_path / "checkpoints", tmp_path / "report.json"
    result = subprocess.run([sys.executable, str(SCRIPT), "verify", "--source-dir", str(tmp_path),
                             "--output-dir", str(output), "--report", str(report)],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert "explicitly trust" in result.stderr
    assert not output.exists() and not report.exists()


def test_capture_does_not_overwrite_incomplete_evidence(tmp_path):
    marker = tmp_path / "previous_full.pt"
    marker.write_bytes(b"preserve incomplete capture")
    result = subprocess.run([sys.executable, str(SCRIPT), "capture", "--source-dir", str(tmp_path),
                             "--output-dir", str(tmp_path)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert "nonempty capture directory" in result.stderr
    assert marker.read_bytes() == b"preserve incomplete capture"
