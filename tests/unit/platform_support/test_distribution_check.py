"""Archive contracts are testable without a GPU or a full FEDOT install."""

from copy import deepcopy
from io import BytesIO
import tarfile
from zipfile import ZipFile

import pytest

from tools.distribution_check import (
    expected_resources,
    inspect_distributions,
    inspect_metadata,
    inspect_sdist,
    inspect_wheel,
    main,
)


@pytest.fixture
def project():
    return {
        "name": "fedot-ind",
        "version": "0.5.0",
        "requires-python": ">=3.10,<3.12",
        "dependencies": ["numpy>=1.24,<2", "fedot @ git+https://example.org/fedot.git@" + "a" * 40],
        "optional-dependencies": {"dev": ["pytest>=7", "tomli>=2; python_version < '3.11'"]},
    }


def metadata_bytes(project, *, replacements=None, extra_lines=()):
    fields = {
        "Metadata-Version": "2.4",
        "Name": project["name"],
        "Version": project["version"],
        "Requires-Python": project["requires-python"],
    }
    fields.update(replacements or {})
    lines = [f"{name}: {value}" for name, value in fields.items()]
    lines += [f"Requires-Dist: {item}" for item in project["dependencies"]]
    lines += [
        "Provides-Extra: dev",
        "Requires-Dist: pytest>=7; extra == 'dev'",
        "Requires-Dist: tomli>=2; python_version < '3.11' and extra == 'dev'",
    ]
    return ("\n".join([*lines, *extra_lines]) + "\n\n").encode()


def wheel_file(tmp_path, raw, *, resources=("fedot_ind/data/defaults.json",), metadata=True, additional=()):
    path = tmp_path / "fedot_ind-0.5.0-py3-none-any.whl"
    with ZipFile(path, "w") as archive:
        if metadata:
            archive.writestr("fedot_ind-0.5.0.dist-info/METADATA", raw)
        archive.writestr("fedot_ind/__init__.py", "__version__ = '0.5.0'")
        for name in resources:
            archive.writestr(name, "{}")
        for name in additional:
            archive.writestr(name, "")
    return path


def sdist_file(tmp_path, raw, *, omit=(), additional=()):
    path = tmp_path / "fedot_ind-0.5.0.tar.gz"
    content = {
        "PKG-INFO": raw,
        "pyproject.toml": b"[project]",
        "setup.py": b"from setuptools import setup\nsetup()\n",
        "README_en.rst": b"README",
        "requirements.txt": b"numpy<2\n",
        "requirements-tensor.txt": b"numpy<2\n",
        "fedot_ind/data/defaults.json": b"{}",
    }
    content.update({name: b"" for name in additional})
    with tarfile.open(path, "w:gz") as archive:
        for name, data in content.items():
            if name in omit:
                continue
            info = tarfile.TarInfo("fedot_ind-0.5.0/" + name)
            info.size = len(data)
            archive.addfile(info, BytesIO(data))
    return path


def test_build_metadata_matches_source_and_optional_markers(project):
    assert inspect_metadata("fixture", metadata_bytes(project), project) == ()


@pytest.mark.parametrize("field,value", [
    ("Name", "another-library"),
    ("Version", "1.0.0"),
    ("Requires-Python", ">=3.9"),
    ("Requires-Python", "broken range"),
])
def test_metadata_mismatch_is_a_structured_failure(project, field, value):
    issues = inspect_metadata("fixture", metadata_bytes(project, replacements={field: value}), project)
    assert any(issue.field == field for issue in issues)


def test_pypi_publication_guard_does_not_reject_local_snapshot_build(project):
    raw = metadata_bytes(project)
    assert not inspect_metadata("fixture", raw, project)
    issues = inspect_metadata("fixture", raw, project, pypi=True)
    assert len(issues) == 1
    assert "PyPI rejects direct URL" in issues[0].message


@pytest.mark.parametrize("additional", [
    "Requires-Dist: numpy>=1.24,<2",
    "Requires-Dist: definitely invalid >>",
    "Requires-Dist: undeclared-package>=1",
])
def test_unexpected_invalid_or_duplicate_requirements(project, additional):
    issues = inspect_metadata("fixture", metadata_bytes(project, extra_lines=[additional]), project)
    assert issues and all(issue.field == "Requires-Dist" for issue in issues)


def test_missing_extra_is_detected(project):
    raw = metadata_bytes(project).replace(b"Provides-Extra: dev\n", b"")
    assert any(issue.field == "Provides-Extra" for issue in inspect_metadata("fixture", raw, project))


def test_missing_runtime_dependency_is_detected(project):
    raw = metadata_bytes(project).replace(b"Requires-Dist: numpy>=1.24,<2\n", b"")
    assert any("Missing:" in issue.message for issue in inspect_metadata("fixture", raw, project))


def test_requirement_marker_order_is_preserved(project):
    changed = deepcopy(project)
    changed["optional-dependencies"]["dev"][1] = "tomli>=2; python_version < '3.10'"
    assert inspect_metadata("fixture", metadata_bytes(project), changed)


def test_wheel_resources_and_payload(project, tmp_path):
    path = wheel_file(tmp_path, metadata_bytes(project))
    assert inspect_wheel(path, project, ["fedot_ind/data/defaults.json"]) == ()
    assert any(issue.field == "package-data" for issue in inspect_wheel(path, project, ["fedot_ind/data/missing.json"]))


@pytest.mark.parametrize("path", ["examples/raw.csv", "benchmark/run.py",
                         "tools/dev.py", ".codex/SKILL.md", "rogue.pth"])
def test_wheel_does_not_ship_developer_tools_or_data(project, tmp_path, path):
    wheel = wheel_file(tmp_path, metadata_bytes(project), additional=[path])
    assert any(issue.field == "contents" for issue in inspect_wheel(wheel, project, []))


def test_wheel_requires_one_metadata_file(project, tmp_path):
    wheel = wheel_file(tmp_path, b"", metadata=False)
    assert inspect_wheel(wheel, project, [])[0].field == "METADATA"
    wheel = wheel_file(tmp_path, metadata_bytes(project), additional=["other.dist-info/METADATA"])
    assert inspect_wheel(wheel, project, [])[0].field == "METADATA"


def test_sdist_is_complete_without_external_data(project, tmp_path):
    source = sdist_file(tmp_path, metadata_bytes(project))
    assert inspect_sdist(source, project, ["fedot_ind/data/defaults.json"]) == ()


@pytest.mark.parametrize("omit,field", [
    ("PKG-INFO", "PKG-INFO"),
    ("setup.py", "package-data"),
    ("README_en.rst", "package-data"),
    ("requirements.txt", "package-data"),
    ("requirements-tensor.txt", "package-data"),
    ("fedot_ind/data/defaults.json", "package-data"),
])
def test_sdist_missing_inputs(project, tmp_path, omit, field):
    source = sdist_file(tmp_path, metadata_bytes(project), omit=[omit])
    assert any(issue.field == field for issue in inspect_sdist(source, project, ["fedot_ind/data/defaults.json"]))


@pytest.mark.parametrize("name", ["examples/data/local.npy", ".codex/SKILL.md",
                         "tools/platform_support/compatibility.json"])
def test_sdist_no_private_or_developer_data(project, tmp_path, name):
    source = sdist_file(tmp_path, metadata_bytes(project), additional=[name])
    assert any(issue.field == "contents" for issue in inspect_sdist(source, project, []))


def test_package_data_glob_is_explicit_and_deterministic(tmp_path):
    data = tmp_path / "fedot_ind/data"
    data.mkdir(parents=True)
    (data / "b.json").write_text("{}")
    (data / "a.json").write_text("{}")
    (data / "private.npy").write_bytes(b"not packaged")
    metadata = {"tool": {"setuptools": {"package-data": {"fedot_ind": ["data/*.json"]}}}}
    assert expected_resources(tmp_path, metadata) == ("fedot_ind/data/a.json", "fedot_ind/data/b.json")
    assert expected_resources(tmp_path, metadata) == expected_resources(tmp_path, metadata)
    metadata["tool"]["setuptools"]["package-data"]["fedot_ind"] = ["data/missing*.json"]
    with pytest.raises(ValueError, match="no files"):
        expected_resources(tmp_path, metadata)


def test_empty_output_directory_is_not_a_success(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fedot-ind'\nversion='0.5.0'\n")
    report = inspect_distributions(tmp_path, tmp_path / "missing")
    assert not report.ok and report.issues[0].field == "archives"


def test_unreadable_project_cli_is_a_json_failure(tmp_path, capsys):
    assert main(["--project", str(tmp_path), "--json"]) == 1
    import json
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert result["issues"][0]["field"] == "project"


def test_archive_inspection_does_not_extract_files(project, tmp_path):
    source = sdist_file(tmp_path, metadata_bytes(project), additional=["../../outside"])
    inspect_sdist(source, project, [])
    assert list(tmp_path.iterdir()) == [source]
