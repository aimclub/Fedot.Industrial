"""Metadata-only environment inspection; no model packages are imported."""

import json
import platform
from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Mapping

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from .schema import Compatibility, FedotProfile, Issue, PythonStatus
from .verification import Project, profile_dependencies


@dataclass(frozen=True)
class InstalledDistribution:
    version: str
    direct_url: object | None = None
    issues: tuple[Issue, ...] = ()


def python_decision(version: str, compatibility: Compatibility) -> dict:
    try:
        parsed = Version(version)
    except InvalidVersion:
        return {"version": version, "status": "invalid", "reason": "Invalid Python version.", "supported": False}
    minor = f"{parsed.major}.{parsed.minor}"
    row = next((row for row in compatibility.python if row.version == minor), None)
    if row is None:
        return {
            "version": version,
            "status": "unsupported",
            "reason": "Python minor version is absent from the policy.",
            "supported": False}
    supported = (row.status is PythonStatus.SUPPORTED and not parsed.is_prerelease
                 and SpecifierSet(compatibility.requires_python).contains(parsed))
    return {
        "version": version,
        "status": row.status.value if supported or row.status is PythonStatus.AUDIT_ONLY else "unsupported",
        "reason": row.reason if not parsed.is_prerelease else "Prerelease Python is not a supported release target.",
        "supported": supported}


def _source_matches(requirement: Requirement, direct_url: object) -> bool:
    if not isinstance(direct_url, dict) or not isinstance(direct_url.get("url"), str):
        return False
    url = requirement.url
    if url.startswith("git+"):
        repository, separator, sha = url[4:].rpartition("@")
        vcs = direct_url.get("vcs_info")
        return bool(separator and isinstance(vcs, dict) and vcs.get("vcs") == "git"
                    and direct_url["url"] == repository and vcs.get("commit_id") == sha)
    return direct_url["url"] == url


def inspect_environment(project: Project, compatibility: Compatibility, profile: FedotProfile, python_version: str,
                        marker_environment: Mapping[str, str], installed: Mapping[str, InstalledDistribution]) -> dict:
    """Interpret an explicit environment snapshot deterministically."""
    decision = python_decision(python_version, compatibility)
    issues: list[Issue] = []
    if not decision["supported"]:
        issues.append(Issue("python", "unsupported", decision["reason"]))
    try:
        matches_project = SpecifierSet(project.requires_python).contains(Version(python_version))
    except InvalidVersion:
        matches_project = False
    if not matches_project:
        issues.append(
            Issue(
                "project.requires-python",
                "incompatible",
                "Running Python does not satisfy project metadata."))
    dependencies = []
    markers = dict(marker_environment, extra="")
    for value in profile_dependencies(project, profile):
        requirement = Requirement(value)
        name = canonicalize_name(requirement.name)
        record = {"name": name, "requirement": value}
        if requirement.marker and not requirement.marker.evaluate(markers):
            dependencies.append(dict(record, status="marker-skipped"))
            continue
        distribution = installed.get(name)
        if distribution is None:
            dependencies.append(dict(record, status="missing"))
            issues.append(Issue(f"dependencies.{name}", "missing", "Distribution metadata is not installed."))
            continue
        record["version"] = distribution.version
        issues.extend(distribution.issues)
        try:
            version = Version(distribution.version)
            compatible = requirement.specifier.contains(
                version, prereleases=True if requirement.url and not requirement.specifier else None)
        except InvalidVersion:
            compatible = False
        if not compatible:
            record["status"] = "incompatible"
            issues.append(
                Issue(
                    f"dependencies.{name}",
                    "incompatible",
                    "Installed version does not satisfy the declared specifier or is invalid."))
        else:
            record["status"] = "installed"
        if requirement.url:
            verified = _source_matches(requirement, distribution.direct_url)
            record["source"] = "verified" if verified else "unverified"
            if not verified:
                issues.append(
                    Issue(
                        f"dependencies.{name}.source",
                        "source-unverified",
                        "PEP 610 metadata does not prove the declared repository and commit; "
                        "installed version alone cannot verify a SHA."))
        dependencies.append(record)
    return {"ok": not issues, "profile": profile.name, "python": decision, "dependencies": dependencies,
            "optional_extras": [name for name, _ in project.optional_dependencies],
            "issues": [asdict(issue) for issue in issues]}


def collect_installed(project: Project, profile: FedotProfile) -> dict[str, InstalledDistribution]:
    installed = {}
    for value in profile_dependencies(project, profile):
        name = canonicalize_name(Requirement(value).name)
        if name in installed:
            continue
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        except (OSError, ValueError) as error:
            installed[name] = InstalledDistribution("", issues=(Issue(f"dependencies.{name}", "metadata", str(error)),))
            continue
        direct_url = None
        issues = []
        try:
            text = distribution.read_text("direct_url.json")
            if text is not None:
                direct_url = json.loads(text)
            version = distribution.version
        except (OSError, ValueError, UnicodeError) as error:
            version = ""
            issues.append(Issue(f"dependencies.{name}", "metadata", str(error)))
        installed[name] = InstalledDistribution(version or "", direct_url, tuple(issues))
    return installed


def current_environment(project: Project, compatibility: Compatibility, profile: FedotProfile) -> dict:
    return inspect_environment(
        project,
        compatibility,
        profile,
        platform.python_version(),
        default_environment(),
        collect_installed(project, profile))
