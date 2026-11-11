"""Unsafe checkpoint reads require an explicit, local trust decision."""
from pathlib import Path
import json
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


def test_current_neural_api_capture_and_verify(tmp_path):
    source = SCRIPT.parents[1]
    output = tmp_path / "checkpoints"
    report = tmp_path / "verification.json"
    common = ["--source-dir", str(source), "--output-dir", str(output)]
    capture = subprocess.run([sys.executable, str(SCRIPT), "capture", *common],
                             capture_output=True, text=True, timeout=60)
    assert capture.returncode == 0, capture.stderr
    verify = subprocess.run([sys.executable, str(SCRIPT), "verify", *common,
                             "--trust-local-checkpoints", "--report", str(report)],
                            capture_output=True, text=True, timeout=60)
    assert verify.returncode == 0, verify.stderr
    result = json.loads(report.read_text())
    assert len(result["cases"]) == 12
    assert sum(case["full_module_restored"] for case in result["cases"]) == 11
    assert all(case["output_max_absolute_error"] == 0 for case in result["cases"])
