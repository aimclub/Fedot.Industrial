import copy
import json
import subprocess
import sys
from dataclasses import replace

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement

from tools.platform_support import parse_compatibility, parse_project, render_requirements, verify_project
from tools.platform_support.__main__ import main
from tools.platform_support.environment import (
    InstalledDistribution, collect_installed, inspect_environment, python_decision,
)
from tools.platform_support.loading import default_root, load_project
from tools.platform_support.verification import EXPORT_HEADER, parse_requirements


@pytest.fixture
def policy_raw():
    path = default_root() / "tools/platform_support/compatibility.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def policy(policy_raw):
    result = parse_compatibility(policy_raw)
    assert result.ok, result.issues
    return result.value


@pytest.fixture
def metadata(policy):
    profile_extras = {
        profile.extra: [f"fedot @ {profile.requirement_url}"]
        for profile in policy.profiles
    }
    return {
        "project": {
            "name": "fixture-project", "version": "1.0.0",
            "requires-python": ">=3.10, <3.12",
            "dependencies": [
                "numpy<2",
                *[
                    item.requirement
                    for item in policy.known_constraints
                    if Requirement(item.requirement).name != "numpy"
                ],
                "sample[feature]>=1; python_version < '3.11'",
            ],
            "optional-dependencies": {**profile_extras, "test": ["pytest>=7"]},
        },
        "build-system": {"requires": ["setuptools"], "build-backend": "setuptools.build_meta"},
    }


@pytest.fixture
def project(metadata):
    result = parse_project(metadata)
    assert result.ok, result.issues
    return result.value


@pytest.fixture
def fixture_root(tmp_path, metadata, project, policy):
    table = metadata["project"]
    text = ("[project]\nname = 'fixture-project'\nversion = '1.0.0'\n"
            f"requires-python = {json.dumps(table['requires-python'])}\n"
            f"dependencies = {json.dumps(table['dependencies'])}\n"
            "[project.optional-dependencies]\n"
            + "".join(f"{name} = {json.dumps(values)}\n"
                      for name, values in table["optional-dependencies"].items())
            +
            "[build-system]\nrequires = ['setuptools']\nbuild-backend = 'setuptools.build_meta'\n")
    (tmp_path / "pyproject.toml").write_text(text, encoding="utf-8")
    for profile in policy.profiles:
        (tmp_path / profile.requirements_file).write_bytes(
            render_requirements(project, profile).encode("utf-8"))
    return tmp_path


def requirements_map(project, policy):
    return {profile.name: render_requirements(project, profile) for profile in policy.profiles}


def test_valid_policy_is_typed_and_preserves_current_planned_distinction(policy):
    assert policy.schema_version == 2
    assert policy.default_profile == "legacy"
    assert policy.current.sha == "42f3ba490407a1106e898e94232f2afd0a78f73f"
    assert policy.planned.sha == "d1875e7a1ba49c94d13c51d97ec78829bf459ca8"
    assert policy.planned.reference == "refactor/fedot_1.0.0"
    assert policy.current.status.value == "supported" and policy.planned.status.value == "experimental"
    assert {profile.extra for profile in policy.profiles} == {"fedot-legacy", "fedot-tensor"}
    assert {item.requirement for item in policy.known_constraints} >= {
        "numpy<2", "dask-ml==2024.4.4", "giotto-tda==0.6.2", "scikit-learn==1.3.2",
    }


def test_policy_accumulates_independent_field_issues(policy_raw):
    policy_raw["surprise"] = 1
    policy_raw["schema_version"] = True
    policy_raw["profiles"]["legacy"]["sha"] = "branch"
    policy_raw["profiles"]["tensor"]["status"] = "current"
    policy_raw["python"][0]["reason"] = None
    policy_raw["python"][1]["status"] = []
    policy_raw["known_constraints"][0]["requirement"] = "bad requirement !!!"
    result = parse_compatibility(policy_raw)
    assert not result.ok and result.value is None
    assert {issue.code for issue in result.issues} >= {"unknown", "version", "sha", "status", "type", "requirement"}


@pytest.mark.parametrize("raw", [None, [], "text", 1, {}, {"python": "3.10"}])
def test_bad_policy_shape_returns_issues_not_exceptions(raw):
    result = parse_compatibility(raw)
    assert not result.ok and result.issues


@pytest.mark.parametrize("sha", ["main", "a" * 39, "a" * 41, "G" * 40, "A" * 40, 1, None])
def test_sha_validation(policy_raw, sha):
    policy_raw["profiles"]["legacy"]["sha"] = sha
    assert any(issue.field == "compatibility.profiles.legacy.sha" for issue in parse_compatibility(policy_raw).issues)


@pytest.mark.parametrize("repository",
                         ["http://host/repo.git",
                          "https://user:secret@host/repo.git",
                          "https://host/repo.git#main",
                          "https://["])
def test_repository_validation(policy_raw, repository):
    policy_raw["profiles"]["legacy"]["repository"] = repository
    assert any(issue.code == "repository" for issue in parse_compatibility(policy_raw).issues)


def test_matrix_complete_and_unique(policy_raw):
    missing = copy.deepcopy(policy_raw)
    missing["python"].pop(1)
    assert any(issue.code == "matrix" for issue in parse_compatibility(missing).issues)
    policy_raw["python"].append(policy_raw["python"][0])
    assert any(issue.code == "duplicate" for issue in parse_compatibility(policy_raw).issues)


@pytest.mark.parametrize("specifier", [">=3.10", "not-python", ">=3.12,<3.10", ">=3.10,<4.0"])
def test_policy_requires_bounded_coherent_range(policy_raw, specifier):
    policy_raw["requires_python"] = specifier
    assert not parse_compatibility(policy_raw).ok


def test_unknown_nested_schema_fields_rejected(policy_raw):
    policy_raw["python"][0]["unknown"] = True
    policy_raw["known_constraints"][0]["unknown"] = True
    policy_raw["profiles"]["legacy"]["unknown"] = True
    result = parse_compatibility(policy_raw)
    assert len([issue for issue in result.issues if issue.code == "unknown"]) == 3


@pytest.mark.parametrize("raw", [None, [], {"project": []},
                         {"project": {"requires-python": "broken", "dependencies": [42, "a !!!"]}}])
def test_malformed_project_returns_structured_issues(raw):
    assert not parse_project(raw).ok


def test_project_accumulates_and_rejects_duplicate_metadata(metadata):
    metadata["tool"] = {"poetry": {"name": "fixture"}}
    metadata["project"]["requires-python"] = "bad"
    metadata["project"]["optional-dependencies"] = {"bad name": ["bad !!!"], "dev_test": [], "dev-test": []}
    metadata["project"]["dynamic"] = ["dependencies"]
    codes = {issue.code for issue in parse_project(metadata).issues}
    assert codes >= {"duplicate-metadata", "specifier", "extra", "requirement", "duplicate", "dynamic"}


@pytest.mark.parametrize("dependencies", [
    ["my_package>=1", "My-Package<2"],
    ["sample[a,b]>=1", "sample[b,a]>=2"],
    ["sample>=1; python_version < '3.11'", 'Sample>=2; python_version < "3.11"'],
])
def test_duplicate_requirements_use_normalized_pep508_identity(dependencies):
    result = parse_requirements(dependencies, "deps")
    assert any(issue.code == "duplicate" for issue in result.issues)


def test_distinct_markers_and_extras_are_not_guessed_or_collapsed():
    requirements = ["sample[a]>=1", "sample[b]>=1", "other; python_version<'3.11'", "other; python_version>='3.11'"]
    assert parse_requirements(requirements, "deps").value == tuple(requirements)


@pytest.mark.parametrize("profile_name", ["legacy", "tensor"])
def test_generated_export_roundtrip_and_order(project, metadata, policy, profile_name):
    profile = policy.profile(profile_name)
    text = render_requirements(project, profile)
    assert text.startswith(EXPORT_HEADER.format(profile=profile_name) + "\n")
    assert text.endswith("\n")
    assert text.splitlines()[1:] == [
        *metadata["project"]["dependencies"],
        *metadata["project"]["optional-dependencies"][profile.extra],
    ]
    parsed = parse_requirements(text.splitlines()[1:], "requirements")
    assert parsed.ok
    assert [str(Requirement(value)) for value in parsed.value] == [
        str(Requirement(value)) for value in project.dependencies + project.extra(profile.extra)]


def test_valid_project_check(project, policy):
    assert verify_project(project, policy, requirements_map(project, policy)) == ()


@pytest.mark.parametrize("replacement", ["numpy>=1", "dask-ml>=2025",
                         "scikit-learn>=1.6", "spacy>=3.5,<3.6; python_version < '3.11'"])
def test_required_profile_bounds_cannot_disappear(project, policy, replacement):
    name = Requirement(replacement).name
    changed = replace(project, dependencies=tuple(
        replacement if Requirement(value).name == name else value for value in project.dependencies))
    issues = verify_project(changed, policy, requirements_map(changed, policy))
    assert any(issue.code == "constraint" for issue in issues)


@pytest.mark.parametrize("dependency", [
    "fedot>=0.7", "fedot @ git+https://github.com/aimclub/FEDOT.git@main",
    "fedot @ git+https://github.com/aimclub/FEDOT.git@27b3f2aa1319cf2eae34d6301357095b8c84fce8",
    "fedot[extra] @ git+https://github.com/aimclub/FEDOT.git@42f3ba490407a1106e898e94232f2afd0a78f73f",
    "fedot @ git+https://github.com/aimclub/FEDOT.git@"
    "42f3ba490407a1106e898e94232f2afd0a78f73f ; python_version < '3.11'",
])
def test_project_rejects_noncurrent_or_conditional_fedot(project, policy, dependency):
    changed = replace(project, optional_dependencies=tuple(
        (name, (dependency,)) if name == policy.current.extra else (name, values)
        for name, values in project.optional_dependencies
    ))
    assert any(issue.code == "sha" for issue in verify_project(
        changed, policy, requirements_map(changed, policy)))


def test_project_rejects_fedot_in_base_dependencies(project, policy):
    changed = replace(project, dependencies=project.dependencies + (
        f"fedot @ {policy.current.requirement_url}",))
    assert any(issue.code == "profile" for issue in verify_project(
        changed, policy, requirements_map(changed, policy)))


def test_project_checks_matrix_and_export_independently(project, policy):
    changed = replace(project, requires_python=">=3.10,<3.13")
    issues = verify_project(changed, policy, {
        profile.name: "numpy<2\nnumpy<2\n" for profile in policy.profiles})
    assert {issue.code for issue in issues} >= {"matrix", "export", "duplicate"}


@pytest.mark.parametrize("version,status,supported", [
    ("3.10.0", "supported", True), ("3.11.12", "supported", True),
    ("3.12.1", "audit-only", False), ("3.9.0", "unsupported", False),
    ("3.13.0", "unsupported", False), ("3.11.0rc1", "unsupported", False),
    ("garbage", "invalid", False),
])
def test_python_decisions(policy, version, status, supported):
    result = python_decision(version, policy)
    assert (result["status"], result["supported"]) == (status, supported)
    assert result["reason"]


def installed_snapshot(policy, numpy="1.26.4"):
    return {"fedot": InstalledDistribution("0.7.5",
                                           {"url": policy.current.repository,
                                            "vcs_info": {"vcs": "git",
                                                         "commit_id": policy.current.sha}}),
            "numpy": InstalledDistribution(numpy),
            "typing": InstalledDistribution("3.7.4.3"),
            "dask-ml": InstalledDistribution("2024.4.4"),
            "giotto-tda": InstalledDistribution("0.6.2"),
            "scikit-learn": InstalledDistribution("1.3.2"),
            "spacy": InstalledDistribution("3.5.4"),
            "sample": InstalledDistribution("1.2"),
            }


def markers(version="3.11"):
    return dict(default_environment(), python_version=version, python_full_version=version + ".1")


def test_environment_metadata_only_success_with_marker_skipped(project, policy):
    snapshot = installed_snapshot(policy)
    snapshot.pop("sample")
    result = inspect_environment(project, policy, policy.current, "3.11.1", markers(), snapshot)
    assert result["ok"] and result["issues"] == []
    rows = {row["name"]: row for row in result["dependencies"]}
    assert rows["fedot"]["source"] == "verified"
    assert rows["sample"]["status"] == "marker-skipped"
    assert result["profile"] == "legacy"
    assert result["optional_extras"] == ["fedot-legacy", "fedot-tensor", "test"]


def test_environment_missing_incompatible_and_unverified_accumulate(project, policy):
    snapshot = installed_snapshot(policy, numpy="2.0")
    snapshot["fedot"] = InstalledDistribution("0.7.5")
    snapshot.pop("sample")
    result = inspect_environment(project, policy, policy.current, "3.10.1", markers("3.10"), snapshot)
    assert not result["ok"]
    assert {issue["code"] for issue in result["issues"]} == {"missing", "incompatible", "source-unverified"}
    statuses = {row["name"]: row["status"] for row in result["dependencies"]}
    assert {name: statuses[name] for name in ("fedot", "numpy", "sample")} == {
        "fedot": "installed", "numpy": "incompatible", "sample": "missing",
    }
    assert next(row for row in result["dependencies"] if row["name"] == "fedot")["source"] == "unverified"


@pytest.mark.parametrize("source",
                         [None,
                          [],
                             {},
                             {"url": "wrong",
                              "vcs_info": {}},
                             {"url": "https://github.com/aimclub/FEDOT.git",
                              "vcs_info": {"vcs": "git",
                                           "commit_id": "main"}}])
def test_installed_fedot_version_does_not_prove_sha(project, policy, source):
    snapshot = installed_snapshot(policy)
    snapshot["fedot"] = InstalledDistribution("0.7.5", source)
    result = inspect_environment(project, policy, policy.current, "3.11.1", markers(), snapshot)
    assert any(issue["code"] == "source-unverified" for issue in result["issues"])


def test_environment_invalid_installed_version_is_structured(project, policy):
    result = inspect_environment(
        project, policy, policy.current, "3.11.1", markers(), installed_snapshot(policy, "not-version"))
    assert any(issue["code"] == "incompatible" for issue in result["issues"])


def test_audit_environment_cannot_pass(project, policy):
    result = inspect_environment(
        project, policy, policy.current, "3.12.1", markers("3.12"), installed_snapshot(policy))
    assert not result["ok"] and result["python"]["status"] == "audit-only"


def test_collection_uses_metadata_boundary_and_retains_errors(project, policy, monkeypatch):
    from importlib import metadata as import_metadata

    class FakeDistribution:
        version = "1.0"

        def read_text(self, name):
            assert name == "direct_url.json"
            return "invalid-json"

    def distribution(name):
        if name == "numpy":
            raise import_metadata.PackageNotFoundError(name)
        return FakeDistribution()

    monkeypatch.setattr(import_metadata, "distribution", distribution)
    result = collect_installed(project, policy.current)
    assert "numpy" not in result
    assert result["fedot"].issues[0].code == "metadata"


def test_cli_json_success_and_failure(fixture_root, capsys):
    assert main(["check", "--root", str(fixture_root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "issues": []}
    (fixture_root / "requirements.txt").write_text("numpy>=2\n", encoding="utf-8")
    assert main(["check", "--root", str(fixture_root), "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert not result["ok"] and any(issue["code"] == "export" for issue in result["issues"])


def test_cli_check_accumulates_missing_files(tmp_path, capsys):
    assert main(["check", "--root", str(tmp_path), "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert len(result["issues"]) == 3
    assert all(issue["code"] == "read" for issue in result["issues"])


@pytest.mark.parametrize("text", ["not [valid toml", "[project]\ndependencies = 1\nrequires-python = 'nonsense'\n"])
def test_loading_invalid_metadata_is_structured(tmp_path, text):
    (tmp_path / "pyproject.toml").write_text(text, encoding="utf-8")
    assert not load_project(tmp_path).ok


@pytest.mark.parametrize("text", ['{"schema_version": 1, "schema_version": 1}', '{broken-json'])
def test_loading_invalid_policy_transport_is_structured(tmp_path, monkeypatch, text):
    from tools.platform_support import loading

    monkeypatch.setattr(loading, "__file__", str(tmp_path / "loading.py"))
    (tmp_path / "compatibility.json").write_text(text, encoding="utf-8")
    result = loading.load_compatibility()
    assert not result.ok and result.issues[0].code == "syntax"


def test_environment_invalid_python_is_structured(project, policy):
    result = inspect_environment(
        project, policy, policy.current, "bad-version", markers(), installed_snapshot(policy))
    assert not result["ok"] and result["python"]["status"] == "invalid"


def test_export_idempotence_and_check_never_writes(fixture_root, capsys):
    path = fixture_root / "requirements.txt"
    tensor_path = fixture_root / "requirements-tensor.txt"
    original = path.read_bytes()
    tensor_original = tensor_path.read_bytes()
    original_stat = path.stat().st_mtime_ns
    assert main(["export", "--root", str(fixture_root), "--check"]) == 0
    assert path.read_bytes() == original and path.stat().st_mtime_ns == original_stat
    assert tensor_path.read_bytes() == tensor_original
    path.write_bytes(b"stale\n")
    stale_stat = path.stat().st_mtime_ns
    assert main(["export", "--root", str(fixture_root), "--check"]) == 1
    assert path.read_bytes() == b"stale\n" and path.stat().st_mtime_ns == stale_stat
    assert main(["export", "--root", str(fixture_root)]) == 0
    assert path.read_bytes() == original
    assert tensor_path.read_bytes() == tensor_original
    assert main(["export", "--root", str(fixture_root)]) == 0
    assert path.read_bytes() == original
    path.unlink()
    assert main(["export", "--root", str(fixture_root), "--check"]) == 1
    assert not path.exists()


def test_export_bad_project_does_not_modify_artifact(fixture_root):
    path = fixture_root / "requirements.txt"
    original = path.read_bytes()
    (fixture_root / "pyproject.toml").write_text("[project]\ndependencies = 1\n", encoding="utf-8")
    assert main(["export", "--root", str(fixture_root)]) == 1
    assert path.read_bytes() == original


def test_cli_environment_boundary_monkeypatch(fixture_root, policy, monkeypatch, capsys):
    from tools.platform_support import environment

    monkeypatch.setattr(environment.platform, "python_version", lambda: "3.11.1")
    monkeypatch.setattr(environment, "default_environment", markers)
    monkeypatch.setattr(environment, "collect_installed", lambda project, profile: installed_snapshot(policy))
    assert main(["environment", "--root", str(fixture_root), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] and result["python"]["version"] == "3.11.1"
    monkeypatch.setattr(environment, "collect_installed", lambda project, profile: {})
    assert main(["environment", "--root", str(fixture_root), "--json"]) == 1
    assert not json.loads(capsys.readouterr().out)["ok"]


def test_module_cli_exit_and_json(fixture_root):
    result = subprocess.run([sys.executable, "-m", "tools.platform_support", "check", "--root",
                            str(fixture_root), "--json"], cwd=default_root(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"]
    (fixture_root / "requirements.txt").unlink()
    failed = subprocess.run([sys.executable, "-m", "tools.platform_support", "check", "--root",
                            str(fixture_root), "--json"], cwd=default_root(), capture_output=True, text=True)
    assert failed.returncode == 1 and not json.loads(failed.stdout)["ok"]


def test_environment_rejects_changed_fedot_policy_before_inspecting_packages(fixture_root, capsys):
    path = fixture_root / "pyproject.toml"
    path.write_text(path.read_text().replace("42f3ba490407a1106e898e94232f2afd0a78f73f", "a" * 40))
    assert main(["environment", "--root", str(fixture_root), "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert any(issue["code"] == "sha" for issue in result["issues"])
    assert "dependencies" not in result
