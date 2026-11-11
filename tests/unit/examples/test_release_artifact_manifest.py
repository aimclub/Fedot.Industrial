"""Release evidence is external; unit fixtures never require local run dumps."""
from hashlib import sha256
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]


def test_release_artifacts_have_safe_unique_paths_and_explicit_availability():
    manifest = json.loads((ROOT / "examples/artifacts/release_1_0_manifest.json").read_text())
    assert manifest["schema_version"] == "industrial_release_artifacts@1"
    assert manifest["external_archive_url"] is None
    assert manifest["publication_status"] == "not_uploaded"
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["source_revision"])
    files = manifest["files"]
    assert len(files) == 8
    assert len({row["local_path"] for row in files}) == len(files)
    for row in files:
        original, local = Path(row["original_path"]), Path(row["local_path"])
        assert not original.is_absolute() and not local.is_absolute()
        assert ".." not in original.parts and ".." not in local.parts
        assert original.parts[:2] == ("docs", "dev_guide")
        if row["role"] == "historical_document":
            assert row["replacement"] == "concise_implementation_and_acceptance_summary"
            assert (ROOT / original).stat().st_size < 16 * 1024
        else:
            assert not (ROOT / original).exists()
        assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
        assert row["bytes"] > 0
        if row["availability"] == "repository":
            assert row["role"] == "frozen_configuration"
            assert local.name == "protocol_20261019.json"
            data = (ROOT / local).read_bytes().replace(b"\r\n", b"\n")
            assert sha256(data).hexdigest() == row["sha256_lf"]
            assert len(data) < 16 * 1024
        else:
            assert row["availability"] == "local_only"
            assert row["role"] in {"experimental_evidence", "historical_document"}
            assert local.parts[:4] == ("examples", "artifacts", "cloud_bundle", "release_1_0_history")


def test_release_archive_catalog_does_not_inherit_an_unverified_public_url(tmp_path):
    from examples.artifacts import write_cloud_bundle_manifest

    catalog = json.loads((ROOT / "examples/artifacts/artifact_catalog.json").read_text())
    group = next(row for row in catalog["groups"] if row["key"] == "release_1_0_history")
    assert group["external_archive_url"] is None
    assert group["publication_status"] == "not_uploaded"
    assert group["storage_policy"] == "manifest_only"
    generated = write_cloud_bundle_manifest(tmp_path / "bundle", repository_root=tmp_path)
    payload = json.loads(generated.read_text())
    row = next(row for row in payload["groups"] if row["key"] == "release_1_0_history")
    assert row["external_archive_url"] is None
    assert row["publication_status"] == "not_uploaded"
