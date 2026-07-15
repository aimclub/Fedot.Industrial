from __future__ import annotations

import json
from pathlib import Path

from examples.artifacts import (
    build_artifact_inventory,
    index_local_artifacts,
    load_artifact_catalog,
    render_artifact_showcase,
    write_cloud_bundle_manifest,
)


def test_artifact_catalog_declares_cloud_handoff_groups() -> None:
    catalog = load_artifact_catalog()
    group_keys = {group["key"] for group in catalog["groups"]}
    showcase_keys = {item["key"] for item in catalog["benchmark_showcase"]}

    assert catalog["version"] == "industrial_examples_artifact_catalog@1"
    assert catalog["external_archive_url"] == "https://disk.yandex.ru/d/Ch_7K26rukpAWw"
    assert catalog["artifact_size_policy"]["max_single_committed_file_mb"] == 5
    assert "benchmark_analysis_publication_packs" in group_keys
    assert "industrial_domain_publication_packs" in group_keys
    assert "examples_utils_local_data" in group_keys
    assert "benchmark_history_archive" in group_keys
    assert "current_kernel_learning_full_runs" in group_keys
    assert {"analysis_uni_clf", "analysis_regr", "m4_analysis", "pipeline_population"} <= showcase_keys


def test_artifact_inventory_is_manifest_driven_and_fast_enough() -> None:
    inventory = build_artifact_inventory()
    by_key = {row.key: row for row in inventory}

    assert by_key["benchmark_analysis_publication_packs"].inventory_mode == "pack"
    assert by_key["benchmark_analysis_publication_packs"].storage_policy == "canonical_local"
    assert by_key["benchmark_analysis_publication_packs"].summary_count >= 1
    assert by_key["benchmark_analysis_publication_packs"].plot_count >= 1
    assert by_key["examples_utils_local_data"].inventory_mode == "shallow"
    assert by_key["examples_utils_local_data"].storage_policy == "manifest_only"
    assert by_key["current_kernel_learning_full_runs"].inventory_mode == "manifest"
    assert by_key["current_kernel_learning_full_runs"].storage_policy == "manifest_only"
    assert by_key["current_kernel_learning_full_runs"].exists_locally


def test_cloud_bundle_indexes_local_lightweight_artifact_files(tmp_path: Path) -> None:
    records = index_local_artifacts(tmp_path / "cloud_bundle")

    local_paths = {record["source_path"] for record in records}

    assert records
    assert any(path.endswith("summary.md") for path in local_paths)
    assert any(path.endswith(".csv") for path in local_paths)
    assert any(path.endswith(".png") for path in local_paths)
    normalized_paths = {path.replace("\\", "/") for path in local_paths}
    assert all(path.startswith("examples/artifacts/cloud_bundle") for path in normalized_paths)
    assert (tmp_path / "cloud_bundle" / "local_artifacts.json").is_file()


def test_artifact_showcase_and_cloud_manifest_render_to_tmp_dir(tmp_path: Path) -> None:
    index_path = render_artifact_showcase(tmp_path / "showcase")
    cloud_manifest_path = write_cloud_bundle_manifest(tmp_path / "cloud_bundle")

    inventory = json.loads((tmp_path / "showcase" / "artifact_inventory.json").read_text(encoding="utf-8"))
    cloud_manifest = json.loads(cloud_manifest_path.read_text(encoding="utf-8"))
    local_artifacts = json.loads((tmp_path / "cloud_bundle" / "local_artifacts.json").read_text(encoding="utf-8"))
    html = index_path.read_text(encoding="utf-8")

    assert index_path.is_file()
    assert (tmp_path / "showcase" / "artifact_inventory.csv").is_file()
    assert "UCR univariate classification" in html
    assert "TSER regression" in html
    assert "M4 Monthly forecasting" in html
    assert any(row["key"] == "benchmark_analysis_publication_packs" for row in inventory)
    assert cloud_manifest["version"] == "industrial_examples_cloud_bundle@1"
    assert cloud_manifest["local_files_manifest"] == "local_artifacts.json"
    assert cloud_manifest["external_archive_url"] == "https://disk.yandex.ru/d/Ch_7K26rukpAWw"
    assert cloud_manifest["artifact_size_policy"]["max_committed_cloud_bundle_mb"] == 100
    assert any(group["category"] == "raw_and_fixture_data" for group in cloud_manifest["groups"])
    assert any(group["local_file_count"] > 0 for group in cloud_manifest["groups"])
    assert local_artifacts
    assert "raw datasets" in cloud_manifest["bundle_policy"]["large_raw_data"].lower()
