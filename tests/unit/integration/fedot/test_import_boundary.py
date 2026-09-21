import ast
import json
from pathlib import Path
import subprocess
import sys

from tools.fedot_import_boundary import LEGACY_MODULES, build_import_map, scan_imports, verify_import_map


ROOT = Path(__file__).resolve().parents[4]
MAP = ROOT / "fedot_ind/integration/fedot/import_map.json"


def test_versioned_import_map_matches_every_legacy_import():
    payload = json.loads(MAP.read_text(encoding="utf-8"))
    assert verify_import_map(ROOT, payload) == ()
    assert payload == build_import_map(scan_imports(ROOT))
    assert {row["legacy"]: row["target"] for row in payload["replacements"]} == LEGACY_MODULES
    assert payload["removal_stage"] == "IND-FEDOT-05"


def test_profile_specific_imports_stay_in_their_adapters():
    package = ROOT / "fedot_ind/integration/fedot"
    observed = {}
    for path in package.glob("*.py"):
        modules = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
        observed[path.name] = modules
    assert all(not any(module.startswith("fedot.") for module in modules)
               for name, modules in observed.items() if name not in {"legacy.py", "tensor.py"})
    assert any(module in LEGACY_MODULES for module in observed["legacy.py"])
    assert "fedot.extensions" in observed["tensor.py"]


def test_public_boundary_import_does_not_load_fedot_runtime():
    script = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r}); import fedot_ind.integration.fedot; "
        "assert not any(name == 'fedot' or name.startswith('fedot.') for name in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_integration_boundary_does_not_create_module_shims():
    source = "\n".join(path.read_text(encoding="utf-8")
                       for path in (ROOT / "fedot_ind/integration/fedot").glob("*.py"))
    assert "sys.modules" not in source
