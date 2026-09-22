"""Prevent metadata, lock, build and CI entrypoints drifting apart."""

import ast
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
import yaml

from tools.distribution_check import expected_resources, tomllib
from tools.platform_support import parse_project, render_requirements, verify_project
from tools.platform_support.loading import load_compatibility


ROOT = Path(__file__).resolve().parents[3]


def project_metadata():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def workflow(name):
    # BaseLoader preserves the YAML key "on" instead of YAML 1.1's boolean coercion.
    return yaml.load((ROOT / ".github/workflows" / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_repository_metadata_export_and_policy_are_coherent():
    parsed = parse_project(project_metadata())
    policy = load_compatibility()
    assert parsed.ok and policy.ok
    texts = {
        profile.name: (ROOT / profile.requirements_file).read_text(encoding="utf-8")
        for profile in policy.value.profiles
    }
    assert all(texts[profile.name] == render_requirements(parsed.value, profile)
               for profile in policy.value.profiles)
    assert verify_project(parsed.value, policy.value, texts) == ()


def test_build_metadata_has_no_second_source():
    raw = project_metadata()
    assert "poetry" not in raw.get("tool", {})
    assert raw["build-system"]["build-backend"] == "setuptools.build_meta"
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert len(calls) == 1 and calls[0].func.id == "setup"
    assert calls[0].args == [] and calls[0].keywords == []
    version_assignment = next(
        node for node in ast.parse((ROOT / "fedot_ind/__init__.py").read_text()).body
        if
        isinstance(node, ast.Assign) and
        any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets))
    assert ast.literal_eval(version_assignment.value) == raw["project"]["version"]


def test_only_runtime_packages_and_declared_resources_are_distributed():
    setuptools = project_metadata()["tool"]["setuptools"]
    assert setuptools["packages"]["find"]["include"] == ["fedot_ind", "fedot_ind.*"]
    assert setuptools["include-package-data"] is False
    paths = expected_resources(ROOT, project_metadata())
    assert len(paths) >= 5
    assert "fedot_ind/integration/fedot/extensions/catalog.json" in paths
    assert all(path.startswith("fedot_ind/") for path in paths)
    assert not any("examples/" in path or "artifacts/" in path for path in paths)


def test_dev_notebook_and_incompatible_research_dependencies_do_not_leak_into_runtime():
    project = project_metadata()["project"]
    runtime_names = {Requirement(value).name.lower() for value in project["dependencies"]}
    assert runtime_names.isdisjoint({"pytest-cov", "build", "uv", "jupyter", "nbconvert", "ipykernel",
                                     "pycaputo", "pymittagleffler", "ripserplusplus"})
    assert {Requirement(value).name.lower()
            for value in project["optional-dependencies"]["dev"]} >= {"pytest", "pytest-cov", "uv"}
    assert {Requirement(value).name.lower()
            for value in project["optional-dependencies"]["notebooks"]} == {"jupyter", "nbconvert", "ipykernel"}


def test_lock_pins_same_fedot_sha_and_python_range():
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    project = project_metadata()["project"]
    policy = load_compatibility().value
    fedot = [package for package in lock["package"] if package["name"] == "fedot"]
    assert len(fedot) == len(policy.profiles)
    sources = {urlsplit(package["source"]["git"]).fragment: urlsplit(package["source"]["git"])
               for package in fedot}
    assert set(sources) == {profile.sha for profile in policy.profiles}
    for profile in policy.profiles:
        source = sources[profile.sha]
        assert source._replace(query="", fragment="").geturl() == profile.repository
        assert parse_qs(source.query) in ({}, {"rev": [profile.sha]})
    assert SpecifierSet(lock["requires-python"]) == SpecifierSet(project["requires-python"])
    local_project = next(package for package in lock["package"] if package["name"] == project["name"])
    assert local_project["version"] == project["version"]
    assert set(local_project["optional-dependencies"]) >= {profile.extra for profile in policy.profiles}


def test_install_matrix_and_python312_audit_are_distinct():
    jobs = workflow("platform_checks.yml")["jobs"]
    matrix = jobs["install"]["strategy"]["matrix"]
    assert matrix["python"] == ["3.10", "3.11"]
    assert set(matrix["os"]) == {"ubuntu-latest", "windows-latest"}
    assert "python312-audit" in jobs
    assert jobs["python312-audit"]["needs"] == "build"
    install_commands = "\n".join(step.get("run", "") for step in jobs["install"]["steps"])
    assert "--frozen" in install_commands and "pip check" in install_commands
    assert "--extra fedot-legacy" in install_commands
    assert "python -I tools/runtime_smoke.py --profile legacy" in install_commands
    assert "poetry" not in install_commands
    tensor = jobs["tensor-integration"]
    assert tensor["strategy"]["matrix"]["python"] == ["3.10", "3.11"]
    tensor_commands = "\n".join(step.get("run", "") for step in tensor["steps"])
    assert "--extra fedot-tensor" in tensor_commands
    assert "environment --profile tensor" in tensor_commands
    assert "tests/unit/integration/fedot" in tensor_commands


def test_unit_and_integration_install_same_locked_profile():
    for name in ("poetry_unit_test.yml", "integration_tests.yml"):
        steps = workflow(name)["jobs"]["test"]["steps"]
        commands = "\n".join(step.get("run", "") for step in steps)
        assert "uv export --frozen --extra fedot-legacy --extra dev" in commands
        assert "pip install --no-deps -e ." in commands
        assert "poetry " not in commands and "pip check" in commands


def test_publication_uses_same_builder_and_pypi_guard():
    for name in ("pypi_release.yml", "python-publish.yml"):
        jobs = workflow(name)["jobs"]
        assert jobs["build"]["uses"] == "./.github/workflows/package_build.yml"
        assert jobs["build"]["with"]["for-pypi"] == "true"
        assert jobs["publish"]["needs"] == "build"
    steps = workflow("package_build.yml")["jobs"]["build"]["steps"]
    assert any(step.get("run") == "python -m build" for step in steps)
    assert any("--pypi" in step.get("run", "") and step.get("if") == "inputs.for-pypi" for step in steps)
