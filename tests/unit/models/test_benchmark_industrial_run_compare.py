from __future__ import annotations

from pathlib import Path

from benchmark.industrial import ModelSpec, compare_registered_runs, load_registry_entries, run_registered_preset


def test_load_registry_entries_reads_registered_runs(tmp_path: Path) -> None:
    output_dir = tmp_path / 'registry_compare'
    run_registered_preset(
        'ucr',
        dataset_name='Lightning7',
        output_dir=output_dir,
        persist_on_run=True,
    )

    frame = load_registry_entries(output_dir)

    assert not frame.empty
    assert 'run_id' in frame.columns
    assert 'task_type' in frame.columns


def test_compare_registered_runs_builds_tables_and_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / 'registry_compare'
    run_registered_preset(
        'ucr',
        dataset_name='Lightning7',
        output_dir=output_dir,
        persist_on_run=True,
    )
    run_registered_preset(
        'ucr',
        dataset_name='Lightning7',
        output_dir=output_dir,
        persist_on_run=True,
        models=(
            ModelSpec(adapter_name='nearest_centroid', display_name='NearestCentroid'),
        ),
    )

    comparison = compare_registered_runs(
        output_dir,
        output_dir=output_dir / 'run_comparison',
    )

    assert not comparison.registry_frame.empty
    assert not comparison.best_models_frame.empty
    assert not comparison.model_metric_frame.empty
    assert comparison.artifact_manifest
    assert any(Path(item.path).exists() for item in comparison.artifact_manifest)
