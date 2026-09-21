"""Pure project metadata and generated requirements checks."""

import re
from dataclasses import dataclass

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name

from .schema import Compatibility, Issue, ParseResult


EXPORT_HEADER = "# Generated from pyproject.toml; run python -m tools.platform_support export."


@dataclass(frozen=True)
class Project:
    requires_python: str
    dependencies: tuple[str, ...]
    optional_dependencies: tuple[tuple[str, tuple[str, ...]], ...]


def parse_requirements(raw: object, path: str) -> ParseResult[tuple[str, ...]]:
    if not isinstance(raw, list):
        return ParseResult(None, (Issue(path, "type", "Expected an array of PEP 508 requirements."),))
    issues: list[Issue] = []
    seen = set()
    values: list[str] = []
    for index, value in enumerate(raw):
        field = f"{path}[{index}]"
        if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
            issues.append(Issue(field, "type", "Expected a single-line requirement string."))
            continue
        try:
            requirement = Requirement(value)
        except InvalidRequirement as error:
            issues.append(Issue(field, "requirement", str(error)))
            continue
        key = (
            canonicalize_name(
                requirement.name), tuple(
                sorted(
                    canonicalize_name(extra) for extra in requirement.extras)), str(
                    requirement.marker) if requirement.marker else "")
        if key in seen:
            issues.append(
                Issue(
                    field,
                    "duplicate",
                    "Duplicate distribution, extras, and marker in this dependency group."))
        seen.add(key)
        values.append(value)
    return ParseResult(None, tuple(issues)) if issues else ParseResult(tuple(values), ())


def parse_project(raw: object) -> ParseResult[Project]:
    if not isinstance(raw, dict):
        return ParseResult(None, (Issue("pyproject", "type", "Expected a TOML table."),))
    issues: list[Issue] = []
    project = raw.get("project")
    if not isinstance(project, dict):
        issues.append(Issue("project", "type", "Expected a PEP 621 project table."))
        project = {}
    requires = project.get("requires-python")
    if not isinstance(requires, str) or not requires.strip():
        issues.append(Issue("project.requires-python", "type", "Expected an explicit Python specifier."))
        requires = ""
    else:
        try:
            SpecifierSet(requires)
        except InvalidSpecifier as error:
            issues.append(Issue("project.requires-python", "specifier", str(error)))
    dependencies = parse_requirements(project.get("dependencies"), "project.dependencies")
    issues.extend(dependencies.issues)
    extras = project.get("optional-dependencies", {})
    optional = []
    if not isinstance(extras, dict):
        issues.append(Issue("project.optional-dependencies", "type", "Expected a table of named dependency arrays."))
    else:
        seen_extras = set()
        for name, values in extras.items():
            path = f"project.optional-dependencies.{name}"
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*", name):
                issues.append(Issue(path, "extra", "Invalid extra name."))
            else:
                normalized = canonicalize_name(name)
                if normalized in seen_extras:
                    issues.append(Issue(path, "duplicate", "Extra names collide after normalization."))
                seen_extras.add(normalized)
            parsed = parse_requirements(values, path)
            issues.extend(parsed.issues)
            if parsed.ok:
                optional.append((name, parsed.value))
    tool = raw.get("tool", {})
    if not isinstance(tool, dict):
        issues.append(Issue("tool", "type", "Expected a table."))
    elif "poetry" in tool:
        issues.append(
            Issue(
                "tool.poetry",
                "duplicate-metadata",
                "Poetry metadata must not duplicate the PEP 621 source."))
    dynamic = project.get("dynamic", [])
    if not isinstance(dynamic, list) or any(not isinstance(item, str) for item in dynamic):
        issues.append(Issue("project.dynamic", "type", "Expected an array of field names."))
    elif {"dependencies", "optional-dependencies", "requires-python"}.intersection(dynamic):
        issues.append(
            Issue(
                "project.dynamic",
                "dynamic",
                "Compatibility metadata must be explicitly declared, not dynamic."))
    if issues:
        return ParseResult(None, tuple(issues))
    return ParseResult(Project(requires, dependencies.value, tuple(optional)), ())


def render_requirements(project: Project) -> str:
    return "\n".join((EXPORT_HEADER, *project.dependencies)) + "\n"


def verify_project(project: Project, compatibility: Compatibility, requirements_text: str) -> tuple[Issue, ...]:
    issues: list[Issue] = []
    if SpecifierSet(project.requires_python) != SpecifierSet(compatibility.requires_python):
        issues.append(Issue("project.requires-python", "matrix", "Python range differs from the compatibility policy."))
    fedot = [Requirement(item) for item in project.dependencies if canonicalize_name(Requirement(item).name) == "fedot"]
    if len(fedot) != 1 or fedot[0].url != compatibility.current.requirement_url or fedot[0].marker or fedot[0].extras:
        issues.append(
            Issue(
                "project.dependencies.fedot",
                "sha",
                "FEDOT must be an unconditional dependency pinned to the current policy repository and full immutable SHA."))
    # Required bounds are explicit policy, not a second dependency resolver.
    for constraint in compatibility.known_constraints:
        required = Requirement(constraint.requirement)
        declared = [Requirement(item) for item in project.dependencies
                    if canonicalize_name(Requirement(item).name) == canonicalize_name(required.name)]
        if (len(declared) != 1 or declared[0].marker or declared[0].url
                or not set(required.specifier).issubset(set(declared[0].specifier))):
            issues.append(Issue(f"project.dependencies.{canonicalize_name(required.name)}", "constraint",
                                f"Required profile bounds are missing: {constraint.requirement}. {constraint.reason}"))
    lines = [line.strip() for line in requirements_text.splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    issues.extend(parse_requirements(lines, "requirements.txt").issues)
    if requirements_text != render_requirements(project):
        issues.append(
            Issue(
                "requirements.txt",
                "export",
                "Requirements differ from the exact generated project.dependencies export."))
    return tuple(issues)
