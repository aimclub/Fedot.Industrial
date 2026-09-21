"""Pure, strict interpretation of the package-local compatibility policy."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar
from urllib.parse import urlsplit

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version


@dataclass(frozen=True)
class Issue:
    field: str
    code: str
    message: str


T = TypeVar("T")


@dataclass(frozen=True)
class ParseResult(Generic[T]):
    value: T | None
    issues: tuple[Issue, ...]

    @property
    def ok(self) -> bool:
        return self.value is not None and not self.issues


class PythonStatus(str, Enum):
    SUPPORTED = "supported"
    AUDIT_ONLY = "audit-only"


@dataclass(frozen=True)
class Target:
    repository: str
    reference: str
    sha: str
    status: str

    @property
    def requirement_url(self) -> str:
        return f"git+{self.repository}@{self.sha}"


@dataclass(frozen=True)
class PythonRow:
    version: str
    status: PythonStatus
    reason: str


@dataclass(frozen=True)
class Constraint:
    requirement: str
    reason: str


@dataclass(frozen=True)
class Compatibility:
    schema_version: int
    requires_python: str
    source: str
    current: Target
    planned: Target
    python: tuple[PythonRow, ...]
    known_constraints: tuple[Constraint, ...]


def _object(raw: object, fields: set[str], path: str, issues: list[Issue]) -> dict:
    if not isinstance(raw, dict):
        issues.append(Issue(path, "type", "Expected an object."))
        return {}
    for key in sorted(fields - raw.keys()):
        issues.append(Issue(f"{path}.{key}", "missing", "Required field is missing."))
    for key in sorted(raw.keys() - fields, key=str):
        issues.append(Issue(f"{path}.{key}", "unknown", "Unknown schema field."))
    return raw


def _string(raw: dict, key: str, path: str, issues: list[Issue]) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        issues.append(Issue(f"{path}.{key}", "type", "Expected a nonempty string."))
        return ""
    return value


def _target(raw: object, path: str, status: str, issues: list[Issue]) -> Target:
    obj = _object(raw, {"repository", "reference", "sha", "status"}, path, issues)
    values = {key: _string(obj, key, path, issues) for key in ("repository", "reference", "sha", "status")}
    repository = values["repository"]
    try:
        url = urlsplit(repository)
        valid_url = (url.scheme == "https" and bool(url.hostname) and url.path.endswith(".git")
                     and not url.username and not url.password and not url.query and not url.fragment)
    except ValueError:
        valid_url = False
    if not valid_url:
        issues.append(
            Issue(
                f"{path}.repository",
                "repository",
                "Expected an HTTPS Git repository URL without credentials."))
    if not re.fullmatch(r"[0-9a-f]{40}", values["sha"]):
        issues.append(Issue(f"{path}.sha", "sha", "Expected a full lowercase 40-character commit SHA."))
    if values["status"] != status:
        issues.append(Issue(f"{path}.status", "status", f"Expected {status!r}."))
    return Target(**values)


def parse_compatibility(raw: object) -> ParseResult[Compatibility]:
    issues: list[Issue] = []
    obj = _object(raw, {"schema_version", "requires_python", "source", "current",
                  "planned", "python", "known_constraints"}, "compatibility", issues)
    if type(obj.get("schema_version")) is not int or obj.get("schema_version") != 1:
        issues.append(Issue("compatibility.schema_version", "version", "Only schema version 1 is supported."))
    source = _string(obj, "source", "compatibility", issues)
    requires = _string(obj, "requires_python", "compatibility", issues)
    try:
        specifier = SpecifierSet(requires)
    except InvalidSpecifier:
        specifier = SpecifierSet()
        issues.append(Issue("compatibility.requires_python", "specifier", "Invalid Python specifier."))
    current = _target(obj.get("current"), "compatibility.current", "current", issues)
    planned = _target(obj.get("planned"), "compatibility.planned", "planned", issues)
    rows: list[PythonRow] = []
    raw_rows = obj.get("python")
    if not isinstance(raw_rows, list) or not raw_rows:
        issues.append(Issue("compatibility.python", "type", "Expected a nonempty array."))
    else:
        seen: set[str] = set()
        for index, raw_row in enumerate(raw_rows):
            path = f"compatibility.python[{index}]"
            row = _object(raw_row, {"version", "status", "reason"}, path, issues)
            version = _string(row, "version", path, issues)
            reason = _string(row, "reason", path, issues)
            status = row.get("status")
            if version in seen:
                issues.append(Issue(f"{path}.version", "duplicate", "Duplicate Python row."))
            seen.add(version)
            if not re.fullmatch(r"\d+\.\d+", version):
                issues.append(Issue(f"{path}.version", "version", "Expected a major.minor Python version."))
                continue
            if not isinstance(status, str) or status not in {item.value for item in PythonStatus}:
                issues.append(Issue(f"{path}.status", "status", "Expected supported or audit-only."))
                continue
            supported = status == PythonStatus.SUPPORTED.value
            if supported != specifier.contains(Version(version)):
                issues.append(Issue(path, "matrix", "Python status disagrees with requires_python."))
            rows.append(PythonRow(version, PythonStatus(status), reason))
        # Version 1 describes a contiguous, bounded range of Python minor versions.
        bounds = {item.operator: item.version for item in specifier}
        if set(bounds) != {">=", "<"} or len(specifier) != 2 or not all(
                re.fullmatch(r"\d+\.\d+", v) for v in bounds.values()):
            issues.append(
                Issue(
                    "compatibility.requires_python",
                    "range",
                    "Expected a bounded minor range: >=major.minor,<major.minor."))
        else:
            lower, upper = Version(bounds[">="]), Version(bounds["<"])
            if lower.major != upper.major or lower.minor >= upper.minor:
                issues.append(
                    Issue(
                        "compatibility.requires_python",
                        "range",
                        "Expected an increasing range within one Python major version."))
            else:
                expected = {f"{lower.major}.{minor}" for minor in range(lower.minor, upper.minor)}
                actual = {row.version for row in rows if row.status is PythonStatus.SUPPORTED}
                if expected != actual:
                    issues.append(
                        Issue(
                            "compatibility.python",
                            "matrix",
                            "Supported rows must cover the entire declared Python range."))
    constraints: list[Constraint] = []
    raw_constraints = obj.get("known_constraints")
    if not isinstance(raw_constraints, list) or not raw_constraints:
        issues.append(Issue("compatibility.known_constraints", "type", "Expected a nonempty array."))
    else:
        for index, raw_constraint in enumerate(raw_constraints):
            path = f"compatibility.known_constraints[{index}]"
            item = _object(raw_constraint, {"requirement", "reason"}, path, issues)
            requirement = _string(item, "requirement", path, issues)
            reason = _string(item, "reason", path, issues)
            try:
                Requirement(requirement)
            except InvalidRequirement as error:
                issues.append(Issue(f"{path}.requirement", "requirement", str(error)))
            constraints.append(Constraint(requirement, reason))
    if issues:
        return ParseResult(None, tuple(issues))
    return ParseResult(Compatibility(1, requires, source, current, planned, tuple(rows), tuple(constraints)), ())
