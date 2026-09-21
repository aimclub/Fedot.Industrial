"""Inspect built archives without importing the ML runtime."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from email.parser import BytesParser
import json
from pathlib import Path
import tarfile
from typing import Mapping, Sequence
from zipfile import BadZipFile, ZipFile

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

try:
    import tomllib
except ImportError:
    import tomli as tomllib


@dataclass(frozen=True)
class ArchiveIssue:
    archive: str
    field: str
    message: str


@dataclass(frozen=True)
class DistributionReport:
    archives: tuple[str, ...]
    issues: tuple[ArchiveIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict:
        return {"ok": self.ok, **asdict(self)}


def expected_resources(root: Path, metadata: Mapping) -> tuple[str, ...]:
    """Expand only declared package data; no repository-wide data discovery."""
    result = set()
    declarations = metadata.get("tool", {}).get("setuptools", {}).get("package-data", {})
    for package, patterns in declarations.items():
        package_path = root.joinpath(*package.split("."))
        for pattern in patterns:
            matches = [path for path in package_path.glob(pattern) if path.is_file()]
            if not matches:
                raise ValueError(f"Package data pattern has no files: {package}/{pattern}")
            result.update(path.relative_to(root).as_posix() for path in matches)
    return tuple(sorted(result))


def _requirements(project: Mapping) -> set[str]:
    result = {str(Requirement(item)) for item in project.get("dependencies", [])}
    for extra, items in project.get("optional-dependencies", {}).items():
        for item in items:
            requirement = Requirement(item)
            marker = f"({requirement.marker}) and extra == '{extra}'" if requirement.marker else f"extra == '{extra}'"
            requirement.marker = Requirement(f"placeholder; {marker}").marker
            result.add(str(requirement))
    return result


def inspect_metadata(
    archive: str, raw: bytes, project: Mapping, *, pypi: bool = False
) -> tuple[ArchiveIssue, ...]:
    metadata = BytesParser().parsebytes(raw)
    issues = []

    def add(field: str, message: str) -> None:
        issues.append(ArchiveIssue(archive, field, message))

    if canonicalize_name(metadata.get("Name", "")) != canonicalize_name(project["name"]):
        add("Name", "Built distribution name differs from pyproject.toml")
    if metadata.get("Version") != project["version"]:
        add("Version", "Built distribution version differs from pyproject.toml")
    try:
        if SpecifierSet(metadata.get("Requires-Python", "")) != SpecifierSet(project["requires-python"]):
            add("Requires-Python", "Python support differs from pyproject.toml")
    except ValueError as error:
        add("Requires-Python", str(error))

    declared = metadata.get_all("Requires-Dist", [])
    parsed = []
    for item in declared:
        try:
            parsed.append(Requirement(item))
        except InvalidRequirement as error:
            add("Requires-Dist", f"Invalid requirement: {error}")
    actual = {str(item) for item in parsed}
    expected = _requirements(project)
    if actual != expected:
        add("Requires-Dist", f"Missing: {sorted(expected - actual)}; unexpected: {sorted(actual - expected)}")
    if len(actual) != len(declared):
        add("Requires-Dist", "Duplicate or invalid built dependencies")
    extras = {canonicalize_name(value) for value in metadata.get_all("Provides-Extra", [])}
    expected_extras = {canonicalize_name(value) for value in project.get("optional-dependencies", {})}
    if extras != expected_extras:
        add("Provides-Extra", "Built extras differ from pyproject.toml")
    if pypi:
        for item in parsed:
            if item.url:
                add("Requires-Dist", f"PyPI rejects direct URL dependency: {item.name}")
    return tuple(issues)


def inspect_wheel(
    path: Path, project: Mapping, resources: Sequence[str], *, pypi: bool = False
) -> tuple[ArchiveIssue, ...]:
    issues = []
    with ZipFile(path) as archive:
        names = archive.namelist()
        metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            return (ArchiveIssue(path.name, "METADATA", "Expected exactly one wheel metadata file"),)
        issues.extend(inspect_metadata(path.name, archive.read(metadata_paths[0]), project, pypi=pypi))
        info_prefix = metadata_paths[0].split("/")[0] + "/"
        forbidden = [name for name in names if not name.startswith(("fedot_ind/", info_prefix))]
        if forbidden:
            issues.append(ArchiveIssue(path.name, "contents", f"Unexpected payload: {sorted(forbidden)}"))
        missing = sorted(set(resources) - set(names))
        if missing:
            issues.append(ArchiveIssue(path.name, "package-data", f"Missing runtime resources: {missing}"))
    return tuple(issues)


def inspect_sdist(
    path: Path, project: Mapping, resources: Sequence[str], *, pypi: bool = False
) -> tuple[ArchiveIssue, ...]:
    with tarfile.open(path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        roots = {name.split("/")[0] for name in members}
        if len(roots) != 1:
            return (ArchiveIssue(path.name, "contents", "Expected a single source distribution root"),)
        prefix = next(iter(roots)) + "/"
        names = {name.removeprefix(prefix) for name in members}
        metadata_path = prefix + "PKG-INFO"
        if metadata_path not in members:
            return (ArchiveIssue(path.name, "PKG-INFO", "Source distribution metadata is missing"),)
        stream = archive.extractfile(members[metadata_path])
        if stream is None:
            return (ArchiveIssue(path.name, "PKG-INFO", "Source distribution metadata is unreadable"),)
        issues = list(inspect_metadata(path.name, stream.read(), project, pypi=pypi))
        source_inputs = {
            "pyproject.toml",
            "setup.py",
            "README_en.rst",
            "requirements.txt",
            "requirements-tensor.txt",
        }
        missing = sorted((set(resources) | source_inputs) - names)
        if missing:
            issues.append(ArchiveIssue(path.name, "package-data", f"Missing source inputs: {missing}"))
        forbidden_roots = {"examples", "benchmark", "tests", "tools", ".codex", ".local", ".git"}
        forbidden = sorted(name for name in names if name.split("/")[0] in forbidden_roots)
        if forbidden:
            issues.append(ArchiveIssue(path.name, "contents", f"Development/data payload in sdist: {forbidden}"))
        return tuple(issues)


def inspect_distributions(root: Path, dist_dir: Path, *, pypi: bool = False) -> DistributionReport:
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    resources = expected_resources(root, metadata)
    wheels = sorted(dist_dir.glob("*.whl"))
    sources = sorted(dist_dir.glob("*.tar.gz"))
    issues = []
    if len(wheels) != 1 or len(sources) != 1:
        issues.append(
            ArchiveIssue(
                str(dist_dir),
                "archives",
                "Expected one wheel and one sdist in a clean output directory"))
    for path, inspector in [(path, inspect_wheel) for path in wheels] + [(path, inspect_sdist) for path in sources]:
        try:
            issues.extend(inspector(path, metadata["project"], resources, pypi=pypi))
        except (BadZipFile, tarfile.TarError, OSError) as error:
            issues.append(ArchiveIssue(path.name, "archive", str(error)))
    return DistributionReport(tuple(path.name for path in wheels + sources), tuple(issues))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--pypi", action="store_true", help="Also reject direct URL dependencies before publication")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = inspect_distributions(args.project, args.dist_dir, pypi=args.pypi)
    except (OSError, ValueError, KeyError) as error:
        report = DistributionReport((), (ArchiveIssue(str(args.project), "project", str(error)),))
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print("Distribution check passed" if report.ok else "Distribution check failed")
        for issue in report.issues:
            print(f"{issue.archive}: {issue.field}: {issue.message}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
