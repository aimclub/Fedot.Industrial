import ast
from pathlib import Path
import subprocess
import sys

from tools.fedot_import_boundary import scan_forbidden_usage


ROOT = Path(__file__).resolve().parents[4]


def test_executable_code_has_no_removed_fedot_paths_or_repository_patches():
    assert scan_forbidden_usage(ROOT) == ()


def test_legacy_integration_modules_and_import_map_are_removed():
    package = ROOT / "fedot_ind/integration/fedot"

    assert not (package / "legacy.py").exists()
    assert not (package / "legacy_repository.py").exists()
    assert not (package / "import_map.json").exists()
    assert not (ROOT / "fedot_ind/core/repository/initializer_industrial_models.py").exists()


def test_fedot_imports_stay_in_effectful_boundary_modules():
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
    effectful_modules = {
        "compatibility.py",
        "detection.py",
        "forecasting.py",
        "temporal_tensor.py",
        "tensor.py",
    }

    assert all(
        not any(module.startswith("fedot.") for module in modules)
        for name, modules in observed.items()
        if name not in effectful_modules
    )
    assert "fedot.core.data.input_data.data" in observed["compatibility.py"]
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


def test_extension_public_index_and_plan_do_not_load_fedot_runtime():
    script = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
        "from fedot_ind.integration.fedot.extensions import build_industrial_extension_plan; "
        "assert build_industrial_extension_plan().operation_names; "
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
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "fedot_ind/integration/fedot").glob("*.py")
    )
    assert "sys.modules" not in source
