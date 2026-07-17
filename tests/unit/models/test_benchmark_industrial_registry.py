from __future__ import annotations

import json
from pathlib import Path

from benchmark.industrial import (
    run_registered_manifest,
    run_registered_manifest_path,
    run_registered_preset,
    run_registered_suite,
)
from benchmark.industrial.experiments.presets import build_local_ucr_suite_config


def _preset_manifest(tmp_path: Path) -> dict:
    return {
        'version': 'benchmark_industrial_manifest@1',
        'kind': 'preset',
        'preset_name': 'm4',
        'subset': 'daily',
        'sample_size': 1,
        'persist_on_run': False,
        'output_dir': str(tmp_path / 'preset_registry_output'),
        'models': [
            {'adapter_name': 'naive_last_value', 'display_name': 'NaiveLastValue'},
        ],
    }


def test_run_registered_manifest_writes_registry_bundle(tmp_path: Path) -> None:
    bundle = run_registered_manifest(_preset_manifest(tmp_path))

    assert bundle.run_dir.exists()
    assert bundle.summary_path.exists()
    assert bundle.registry_entry_path.exists()
    assert bundle.registry_index_path.exists()
    assert (bundle.run_dir / 'resolved_manifest.json').exists()
    assert (bundle.run_dir / 'input_payload.json').exists()
    assert (bundle.run_dir / 'artifact_manifest.json').exists()

    entry = json.loads(bundle.registry_entry_path.read_text(encoding='utf-8'))
    assert entry['execution_mode'] == 'manifest'
    assert entry['run_id'] == bundle.result.run_id


def test_run_registered_preset_writes_registry_bundle(tmp_path: Path) -> None:
    bundle = run_registered_preset(
        'ucr',
        dataset_name='Lightning7',
        output_dir=tmp_path / 'preset_registered_suite',
        persist_on_run=False,
    )

    assert bundle.run_dir.exists()
    assert bundle.registry_entry_path.exists()
    entry = json.loads(bundle.registry_entry_path.read_text(encoding='utf-8'))
    assert entry['execution_mode'] == 'preset'


def test_run_registered_manifest_path_writes_registry_index_files(tmp_path: Path) -> None:
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(_preset_manifest(tmp_path), ensure_ascii=False, indent=2), encoding='utf-8')

    bundle = run_registered_manifest_path(manifest_path)

    registry_dir = bundle.registry_entry_path.parent
    assert (registry_dir / 'run_registry.csv').exists()
    assert (registry_dir / 'run_registry.md').exists()


def test_run_registered_suite_writes_resolved_config(tmp_path: Path) -> None:
    config = build_local_ucr_suite_config(
        dataset_name='Lightning7',
        output_dir=tmp_path / 'registered_suite',
        persist_on_run=False,
    )

    bundle = run_registered_suite(config)

    assert bundle.run_dir.exists()
    assert (bundle.run_dir / 'resolved_config.json').exists()
    summary = json.loads(bundle.summary_path.read_text(encoding='utf-8'))
    assert summary['execution_mode'] == 'suite'
