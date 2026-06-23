from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from benchmark.results.showcase import (
    build_group_result_frame,
    load_showcase_manifest,
    render_results_showcase,
)


def test_default_showcase_manifest_declares_kernel_learning_and_legacy_sources() -> None:
    manifest = load_showcase_manifest()
    groups = {group.key: group for group in manifest.groups}

    assert {"ucr_classification", "tser_regression", "m4_forecasting"} <= set(groups)
    assert any(
        source.path.startswith("benchmark/results/v2_kernel_learning")
        for group in manifest.groups
        for source in group.sources
    )
    assert any(source.role == "sota_reference" for group in manifest.groups for source in group.sources)
    assert manifest.archive_candidates


def test_showcase_renderer_builds_canonical_tables_from_manifest(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    pd.DataFrame(
        [
            {"dataset_name": "D1", "SOTA": 0.86, "IndustrialOld": 0.8},
            {"dataset_name": "D2", "SOTA": 0.81, "IndustrialOld": 0.78},
        ]
    ).to_csv(project_root / "legacy.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset_name": "D1",
                "model_name": "IndustrialNew",
                "metric_name": "accuracy",
                "metric_value": 0.9,
            },
            {
                "dataset_name": "D2",
                "model_name": "IndustrialNew",
                "metric_name": "accuracy",
                "metric_value": 0.79,
            },
        ]
    ).to_csv(project_root / "current.csv", index=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": "unit@1",
                "title": "Unit Showcase",
                "description": "Synthetic showcase",
                "groups": [
                    {
                        "key": "classification",
                        "title": "Classification",
                        "task_type": "ts_classification",
                        "metric_name": "accuracy",
                        "metric_direction": "higher",
                        "target_source_labels": ["current"],
                        "reference_source_labels": ["legacy"],
                        "sources": [
                            {
                                "key": "legacy",
                                "kind": "table",
                                "path": "legacy.csv",
                                "source_label": "legacy",
                                "role": "sota_reference",
                            },
                            {
                                "key": "current",
                                "kind": "table",
                                "path": "current.csv",
                                "source_label": "current",
                                "role": "current_industrial",
                            },
                        ],
                    }
                ],
                "archive_candidates": [
                    {
                        "path": "old",
                        "reason": "superseded",
                        "recommended_action": "archive",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_showcase_manifest(manifest_path)
    normalized = build_group_result_frame(manifest.groups[0], project_root=project_root)
    resolved_path = render_results_showcase(
        tmp_path / "showcase",
        manifest_path=manifest_path,
        project_root=project_root,
    )

    overview = pd.read_csv(tmp_path / "showcase" / "tables" / "benchmark_overview.csv")
    current_best = pd.read_csv(tmp_path / "showcase" / "tables" / "current_best_per_dataset.csv")
    inventory = pd.read_csv(tmp_path / "showcase" / "tables" / "source_inventory.csv")

    assert set(normalized["source_label"]) == {"legacy", "current"}
    assert resolved_path.is_file()
    assert overview.loc[0, "dataset_count"] == 2
    assert set(current_best["best_model"]) == {"IndustrialNew"}
    assert inventory["exists_locally"].all()
    assert (tmp_path / "showcase" / "classification" / "summary.md").is_file()
