"""Formatting must not rewrite runtime guards or local function calls."""
from pathlib import Path
import shlex

import yaml


def test_formatter_does_not_enable_semantic_python_migrations():
    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load((root / ".github/workflows/pep8_action.yml").read_text(encoding="utf-8"))
    step = next(item for item in workflow["jobs"]["autopep8"]["steps"] if item.get("id") == "autopep8")
    args = shlex.split(step["with"]["args"])
    assert "--experimental" not in args
    assert not any(arg.startswith("--aggressive") for arg in args)
    assert "--ignore" in args
    assert "E721" in args[args.index("--ignore") + 1].split(",")
