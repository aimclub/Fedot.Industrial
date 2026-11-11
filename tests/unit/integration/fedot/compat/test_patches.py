from copy import deepcopy
import json
import subprocess

import pytest

from fedot_ind.integration.fedot.compat import patches


def manifest():
    return {"schema_version": "industrial_fedot_compatibility@1", "base_revision": "a" * 40,
            "patches": [{"name": "fix", "resource": "patches/fix.patch", "sha256": "a" * 64,
                         "sources": {"fedot/source.py": "b" * 64}}]}


@pytest.mark.parametrize("change", [
    lambda m: m.update(schema_version="unknown"),
    lambda m: m.update(extra=True),
    lambda m: m.update(base_revision="main"),
    lambda m: m.update(patches=[]),
    lambda m: m["patches"][0].update(resource="../escape.patch"),
    lambda m: m["patches"][0].update(resource="C:/escape.patch"),
    lambda m: m["patches"][0].update(sha256="invalid"),
    lambda m: m["patches"][0].update(sources={"../escape.py": "b" * 64}),
    lambda m: m["patches"].append(deepcopy(m["patches"][0])),
])
def test_manifest_fails_closed(change):
    record = manifest()
    change(record)
    with pytest.raises(ValueError):
        patches.parse_bundle(record)


def test_manifest_parsing_preserves_input_and_line_endings_do_not_change_hash():
    record = manifest()
    before = deepcopy(record)
    bundle = patches.parse_bundle(record)
    assert bundle.patches[0].name == "fix"
    assert record == before
    assert patches.source_digest("a\r\nb\r\n") == patches.source_digest("a\nb\n")


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture
def worktree(tmp_path, monkeypatch):
    root, resources = tmp_path / "fedot", tmp_path / "resources"
    (root / "fedot").mkdir(parents=True)
    resources.mkdir()
    (root / "fedot/__init__.py").write_text("")
    (root / "fedot/source.py").write_text("old\n")
    git(root, "init")
    git(root, "add", "fedot")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    content = ("diff --git a/fedot/source.py b/fedot/source.py\n"
               "--- a/fedot/source.py\n+++ b/fedot/source.py\n@@ -1 +1 @@\n-old\n+new\n")
    (resources / "fix.patch").write_text(content, encoding="utf-8")
    row = {"name": "fix", "resource": "fix.patch", "sha256": patches.source_digest(content),
           "sources": {"fedot/source.py": patches.source_digest("new\n")}}
    bundle = patches.parse_bundle({"schema_version": "industrial_fedot_compatibility@1",
                                   "base_revision": git(root, "rev-parse", "HEAD"), "patches": [row]})
    monkeypatch.setattr(patches, "load_bundle", lambda: bundle)
    monkeypatch.setattr(patches, "files", lambda _: resources)
    return root, resources


def test_application_requires_explicit_execution_and_audit_detects_drift(worktree):
    root, _ = worktree
    assert patches.apply_compatibility_patches(root)["status"] == "planned"
    assert (root / "fedot/source.py").read_text() == "old\n"
    result = patches.apply_compatibility_patches(root, execute=True)
    assert result["status"] == "applied"
    assert patches.audit_compatibility_worktree(root)["base_revision"] == result["base_revision"]
    with pytest.raises(ValueError, match="changed source"):
        patches.apply_compatibility_patches(root, execute=True)
    (root / "fedot/source.py").write_text("unrelated\n")
    with pytest.raises(ValueError, match="source mismatch"):
        patches.audit_compatibility_worktree(root)


def test_modified_source_is_not_overwritten(worktree):
    root, _ = worktree
    (root / "fedot/source.py").write_text("user change\n")
    with pytest.raises(ValueError, match="changed source"):
        patches.apply_compatibility_patches(root, execute=True)
    assert (root / "fedot/source.py").read_text() == "user change\n"


def test_changed_patch_resource_is_not_applied(worktree):
    root, resources = worktree
    (resources / "fix.patch").write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        patches.apply_compatibility_patches(root, execute=True)
    assert (root / "fedot/source.py").read_text() == "old\n"


def test_wrong_revision_is_rejected(worktree):
    root, _ = worktree
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--allow-empty", "-m", "next")
    with pytest.raises(ValueError, match="pinned compatibility base"):
        patches.apply_compatibility_patches(root)


def test_packaged_manifest_and_patch_resources_have_matching_hashes():
    bundle = patches.load_bundle()
    for patch in bundle.patches:
        resource = patches.files(patches.__package__).joinpath(patch.resource)
        assert patches.source_digest(resource.read_text(encoding="utf-8")) == patch.digest
    assert len(bundle.patches) == 4


def test_cli_has_explicit_apply_and_audit_modes(worktree, capsys):
    from fedot_ind.integration.fedot.compat.__main__ import main

    root, _ = worktree
    assert main(["--worktree", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "planned"
    assert main(["--worktree", str(root), "--execute"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "applied"
    assert main(["--worktree", str(root), "--audit"]) == 0
    assert json.loads(capsys.readouterr().out)["patches"][0]["name"] == "fix"
    with pytest.raises(SystemExit):
        main(["--worktree", str(root), "--audit", "--execute"])


def test_nested_worktree_path_is_rejected(worktree):
    root, _ = worktree
    with pytest.raises(ValueError, match="worktree root"):
        patches.apply_compatibility_patches(root / "fedot")
