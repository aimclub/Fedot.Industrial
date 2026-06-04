from __future__ import annotations

from pathlib import Path

from benchmark.industrial import (
    TaskType,
    build_suite_config_from_manifest,
    load_manifest,
    render_resolved_manifest,
    run_manifest,
    run_manifest_path,
    write_example_manifest,
)


def _preset_manifest(tmp_path: Path) -> dict:
    return {
        'version': 'benchmark_industrial_manifest@1',
        'kind': 'preset',
        'preset_name': 'm4',
        'subset': 'daily',
        'sample_size': 1,
        'persist_on_run': False,
        'output_dir': str(tmp_path / 'preset_output'),
        'models': [
            {'adapter_name': 'naive_last_value', 'display_name': 'NaiveLastValue'},
        ],
    }


def _suite_manifest(tmp_path: Path) -> dict:
    return {
        'version': 'benchmark_industrial_manifest@1',
        'kind': 'suite',
        'task_type': 'ts_regression',
        'datasets': [
            {
                'benchmark': 'in_memory_tser',
                'dataset_name': 'toy_tser_manifest',
                'adapter_options': {
                    'record': {
                        'train_features': [[0.0], [1.0], [2.0], [3.0]],
                        'train_target': [1.0, 3.0, 5.0, 7.0],
                        'test_features': [[4.0], [5.0]],
                        'test_target': [9.0, 11.0],
                    }
                },
            }
        ],
        'models': [
            {'adapter_name': 'linear_regressor', 'display_name': 'LinearRegressor'},
        ],
        'artifact_spec': {
            'output_dir': str(tmp_path / 'suite_output'),
            'persist_on_run': False,
        },
        'run_spec': {
            'run_name': 'toy_suite_manifest',
            'primary_metric': 'rmse',
        },
        'metrics': ['rmse', 'mae', 'r2'],
    }


def test_load_manifest_and_build_suite_config_from_yaml(tmp_path: Path) -> None:
    manifest_path = tmp_path / 'suite_manifest.yaml'
    write_example_manifest(manifest_path, _suite_manifest(tmp_path))

    payload = load_manifest(manifest_path)
    config = build_suite_config_from_manifest(payload)

    assert config.task_type is TaskType.TS_REGRESSION
    assert config.datasets[0].benchmark == 'in_memory_tser'
    assert config.models[0].adapter_name == 'linear_regressor'


def test_run_manifest_from_preset_payload() -> None:
    result = run_manifest(_preset_manifest(Path('.')))

    assert result.config.task_type is TaskType.FORECASTING
    assert any(record.status.value == 'success' for record in result.run_records)


def test_run_manifest_path_from_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / 'preset_manifest.json'
    write_example_manifest(manifest_path, _preset_manifest(tmp_path))

    result = run_manifest_path(manifest_path)

    assert result.config.task_type is TaskType.FORECASTING
    assert any(record.status.value == 'success' for record in result.run_records)


def test_render_resolved_manifest_for_suite(tmp_path: Path) -> None:
    resolved = render_resolved_manifest(_suite_manifest(tmp_path))

    assert resolved['task_type'] == 'ts_regression'
    assert resolved['datasets'][0]['dataset_name'] == 'toy_tser_manifest'
    assert resolved['metrics'] == ['rmse', 'mae', 'r2']
