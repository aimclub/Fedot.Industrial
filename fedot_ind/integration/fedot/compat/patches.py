"""Verified patch application to an explicitly selected FEDOT Git worktree."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
import json
from pathlib import Path, PurePosixPath
import subprocess


@dataclass(frozen=True)
class CompatibilityPatch:
    name: str
    resource: str
    digest: str
    source_digests: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CompatibilityBundle:
    base_revision: str
    patches: tuple[CompatibilityPatch, ...]


def source_digest(content: str) -> str:
    """Git treats line endings as transport, not a different source revision."""
    return sha256(content.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def _relative_path(value: object) -> str:
    if (not isinstance(value, str) or not value or "\\" in value
            or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts
            or ":" in value):
        raise ValueError("Compatibility paths must be relative, without parent traversal")
    return value


def _digest(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected a SHA256 digest")
    return value


def parse_bundle(payload: object) -> CompatibilityBundle:
    if (not isinstance(payload, dict) or set(payload) != {"schema_version", "base_revision", "patches"}
            or payload["schema_version"] != "industrial_fedot_compatibility@1"):
        raise ValueError("Invalid compatibility manifest")
    revision = payload["base_revision"]
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Expected a pinned FEDOT commit")
    if not isinstance(payload["patches"], list) or not payload["patches"]:
        raise ValueError("Compatibility manifest must contain patches")
    patches, names, resources, source_paths = [], set(), set(), set()
    for row in payload["patches"]:
        if not isinstance(row, dict) or set(row) != {"name", "resource", "sha256", "sources"}:
            raise ValueError("Invalid compatibility patch record")
        name = row["name"]
        resource = _relative_path(row["resource"])
        if not isinstance(name, str) or not name.strip() or name in names or resource in resources:
            raise ValueError("Compatibility patch names and resources must be unique")
        if not isinstance(row["sources"], dict) or not row["sources"]:
            raise ValueError("Patch must declare its resulting source hashes")
        sources = []
        for path, digest in row["sources"].items():
            path = _relative_path(path)
            if path in source_paths:
                raise ValueError("Patch sources must not overlap")
            source_paths.add(path)
            sources.append((path, _digest(digest)))
        names.add(name)
        resources.add(resource)
        patches.append(CompatibilityPatch(name, resource, _digest(row["sha256"]), tuple(sources)))
    return CompatibilityBundle(revision, tuple(patches))


def load_bundle() -> CompatibilityBundle:
    return parse_bundle(json.loads(files(__package__).joinpath("manifest.json").read_text(encoding="utf-8")))


def _git(root: Path, *arguments: str, input_text: str | None = None) -> str:
    result = subprocess.run(["git", "-C", str(root), *arguments], input=input_text,
                            capture_output=True, text=True, encoding="utf-8", check=False)
    if result.returncode:
        raise ValueError(f"FEDOT compatibility check failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


def _worktree(path: str | Path, bundle: CompatibilityBundle) -> Path:
    root = Path(path).resolve(strict=True)
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("Select the FEDOT worktree root, not a nested directory")
    if not (root / "fedot" / "__init__.py").is_file():
        raise ValueError("Selected worktree is not FEDOT")
    if _git(root, "rev-parse", "HEAD") != bundle.base_revision:
        raise ValueError("FEDOT revision differs from the pinned compatibility base")
    return root


def _source_path(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Compatibility source must stay inside the selected worktree")
    return target


def audit_compatibility_worktree(path: str | Path) -> dict[str, object]:
    """Reject changed patched files; return provenance without changing a checkout."""
    bundle = load_bundle()
    root = _worktree(path, bundle)
    records = []
    for patch in bundle.patches:
        content = files(__package__).joinpath(patch.resource).read_text(encoding="utf-8")
        if source_digest(content) != patch.digest:
            raise ValueError(f"Compatibility patch checksum mismatch: {patch.name}")
        for relative, digest in patch.source_digests:
            target = _source_path(root, relative)
            if not target.is_file() or source_digest(target.read_text(encoding="utf-8")) != digest:
                raise ValueError(f"Compatibility source mismatch: {relative}")
        records.append({"name": patch.name, "sha256": patch.digest,
                        "sources": dict(patch.source_digests)})
    return {"schema_version": "industrial_fedot_compatibility_audit@1",
            "worktree": str(root), "base_revision": bundle.base_revision, "patches": records}


def apply_compatibility_patches(path: str | Path, *, execute: bool = False) -> dict[str, object]:
    """Dry-run by default. No site-packages edits or import-time monkeypatches."""
    bundle = load_bundle()
    root = _worktree(path, bundle)
    contents = []
    for patch in bundle.patches:
        content = files(__package__).joinpath(patch.resource).read_text(encoding="utf-8")
        if source_digest(content) != patch.digest:
            raise ValueError(f"Compatibility patch checksum mismatch: {patch.name}")
        for relative, _ in patch.source_digests:
            target = _source_path(root, relative)
            if _git(root, "status", "--porcelain", "--", relative):
                raise ValueError(f"Compatibility refuses to overwrite a changed source: {relative}")
            # Git reports ignored untracked files differently; never overwrite them.
            tracked = _git(root, "ls-files", "--", relative)
            if target.exists() and not tracked:
                raise ValueError(f"Compatibility refuses to overwrite an untracked source: {relative}")
        contents.append(content)
    combined = "\n".join(contents)
    _git(root, "apply", "--check", "-", input_text=combined)
    if not execute:
        return {"status": "planned", "base_revision": bundle.base_revision,
                "worktree": str(root), "patches": [patch.name for patch in bundle.patches]}
    _git(root, "apply", "-", input_text=combined)
    return {"status": "applied", **audit_compatibility_worktree(root)}
