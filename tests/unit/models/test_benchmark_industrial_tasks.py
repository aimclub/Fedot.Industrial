from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmark.industrial import (
    ArtifactSpec,
    BenchmarkSuiteConfig,
    DatasetSpec,
    ModelSpec,
    RunSpec,
    RunStatus,
    TaskType,
    build_tsc_publication_pack,
    build_tser_publication_pack,
    run_tsc_benchmark_suite,
    run_tser_benchmark_suite,
)
from benchmark.industrial.classification import build_classification_dataset_adapter
from benchmark.industrial.regression import build_regression_dataset_adapter


def _classification_record() -> dict:
    train_features = np.array(
        [
            [0.0, 0.1],
            [0.2, 0.0],
            [1.0, 1.1],
            [0.9, 1.0],
        ]
    )
    train_target = np.array(['a', 'a', 'b', 'b'])
    test_features = np.array(
        [
            [0.1, 0.2],
            [1.1, 0.9],
        ]
    )
    test_target = np.array(['a', 'b'])
    return {
        'train_features': train_features,
        'train_target': train_target,
        'test_features': test_features,
        'test_target': test_target,
    }


def _regression_record() -> dict:
    train_features = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    train_target = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    test_features = np.array([[5.0], [6.0]])
    test_target = np.array([11.0, 13.0])
    return {
        'train_features': train_features,
        'train_target': train_target,
        'test_features': test_features,
        'test_target': test_target,
    }


def test_tsc_suite_runs_and_writes_artifacts(tmp_path: Path) -> None:
    config = BenchmarkSuiteConfig(
        task_type=TaskType.TS_CLASSIFICATION,
        datasets=(
            DatasetSpec(
                benchmark='in_memory_tsc',
                dataset_name='toy_tsc',
                adapter_options={'record': _classification_record()},
            ),
        ),
        models=(
            ModelSpec(adapter_name='majority_class',
                      display_name='MajorityClass'),
            ModelSpec(adapter_name='nearest_centroid',
                      display_name='NearestCentroid'),
        ),
        metrics=('accuracy', 'balanced_accuracy', 'f1_macro'),
        artifact_spec=ArtifactSpec(
            output_dir=str(tmp_path), persist_on_run=True),
        run_spec=RunSpec(run_name='toy_tsc', primary_metric='accuracy'),
    )

    result = run_tsc_benchmark_suite(config)

    assert any(record.status is RunStatus.SUCCESS for record in result.run_records)
    assert any(record.model_name ==
               'NearestCentroid' for record in result.run_records)
    assert result.artifact_manifest
    assert any(Path(item.path).exists() for item in result.artifact_manifest)


def test_tsc_publication_pack_builder_is_public(tmp_path: Path) -> None:
    config = BenchmarkSuiteConfig(
        task_type=TaskType.TS_CLASSIFICATION,
        datasets=(
            DatasetSpec(
                benchmark='in_memory_tsc',
                dataset_name='toy_tsc',
                adapter_options={'record': _classification_record()},
            ),
        ),
        models=(ModelSpec(adapter_name='nearest_centroid',
                display_name='NearestCentroid'),),
        metrics=('accuracy', 'balanced_accuracy', 'f1_macro'),
        artifact_spec=ArtifactSpec(
            output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(run_name='toy_tsc_public_pack',
                         primary_metric='accuracy'),
    )

    result = run_tsc_benchmark_suite(config)
    manifest = build_tsc_publication_pack(
        result, output_dir=tmp_path / 'tsc_public_pack')

    assert manifest
    assert any(Path(item.path).exists() for item in manifest)


def test_local_tsc_adapter_reads_real_tsv_dataset() -> None:
    spec = DatasetSpec(
        benchmark='ucr',
        dataset_name='Lightning7',
    )

    adapter = build_classification_dataset_adapter(spec)
    records = adapter.load_dataset(spec)

    assert len(records) == 1
    record = records[0]
    assert record.metadata['split_provenance'] == 'local_tsv'
    assert record.metadata['source_train_file'] == 'Lightning7_TRAIN.tsv'
    assert len(record.train_features) > 0
    assert len(record.test_features) > 0


def test_local_tsc_adapter_reads_real_ts_dataset() -> None:
    spec = DatasetSpec(
        benchmark='ucr',
        dataset_name='BasicMotions',
    )

    adapter = build_classification_dataset_adapter(spec)
    records = adapter.load_dataset(spec)

    assert len(records) == 1
    record = records[0]
    assert record.metadata['split_provenance'] == 'local_ts'
    assert record.metadata['source_train_file'] == 'BasicMotions_TRAIN.ts'
    assert record.metadata['dimensions'] == 6
    assert len(record.train_features[0]) == 600


def test_tser_suite_runs_and_writes_artifacts(tmp_path: Path) -> None:
    config = BenchmarkSuiteConfig(
        task_type=TaskType.TS_REGRESSION,
        datasets=(
            DatasetSpec(
                benchmark='in_memory_tser',
                dataset_name='toy_tser',
                adapter_options={'record': _regression_record()},
            ),
        ),
        models=(
            ModelSpec(adapter_name='mean_regressor',
                      display_name='MeanRegressor'),
            ModelSpec(adapter_name='linear_regressor',
                      display_name='LinearRegressor'),
        ),
        metrics=('rmse', 'mae', 'r2'),
        artifact_spec=ArtifactSpec(
            output_dir=str(tmp_path), persist_on_run=True),
        run_spec=RunSpec(run_name='toy_tser', primary_metric='rmse'),
    )

    result = run_tser_benchmark_suite(config)

    assert any(record.status is RunStatus.SUCCESS for record in result.run_records)
    assert any(record.model_name ==
               'LinearRegressor' for record in result.run_records)
    assert result.artifact_manifest
    assert any(Path(item.path).exists() for item in result.artifact_manifest)


def test_tser_publication_pack_builder_is_public(tmp_path: Path) -> None:
    config = BenchmarkSuiteConfig(
        task_type=TaskType.TS_REGRESSION,
        datasets=(
            DatasetSpec(
                benchmark='in_memory_tser',
                dataset_name='toy_tser',
                adapter_options={'record': _regression_record()},
            ),
        ),
        models=(ModelSpec(adapter_name='linear_regressor',
                display_name='LinearRegressor'),),
        metrics=('rmse', 'mae', 'r2'),
        artifact_spec=ArtifactSpec(
            output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(run_name='toy_tser_public_pack',
                         primary_metric='rmse'),
    )

    result = run_tser_benchmark_suite(config)
    manifest = build_tser_publication_pack(
        result, output_dir=tmp_path / 'tser_public_pack')

    assert manifest
    assert any(Path(item.path).exists() for item in manifest)


def test_local_tser_adapter_reads_real_ts_dataset() -> None:
    spec = DatasetSpec(
        benchmark='local_tser',
        dataset_name='NaturalGasPricesSentiment',
    )

    adapter = build_regression_dataset_adapter(spec)
    records = adapter.load_dataset(spec)

    assert len(records) == 1
    record = records[0]
    assert record.metadata['split_provenance'] == 'local_ts'
    assert record.metadata['target_type'] == 'regression'
    assert record.metadata['source_train_file'] == 'NaturalGasPricesSentiment_TRAIN.ts'
    assert len(record.train_features[0]) == 20


def test_local_tser_adapter_reads_real_ts_dataset_from_foreign_cwd(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    spec = DatasetSpec(
        benchmark='local_tser',
        dataset_name='NaturalGasPricesSentiment',
    )

    adapter = build_regression_dataset_adapter(spec)
    records = adapter.load_dataset(spec)

    assert len(records) == 1
    assert records[0].metadata['split_provenance'] == 'local_ts'


def test_tser_suite_runs_on_real_local_dataset(tmp_path: Path) -> None:
    config = BenchmarkSuiteConfig(
        task_type=TaskType.TS_REGRESSION,
        datasets=(
            DatasetSpec(
                benchmark='local_tser',
                dataset_name='NaturalGasPricesSentiment',
            ),
        ),
        models=(
            ModelSpec(adapter_name='mean_regressor',
                      display_name='MeanRegressor'),
            ModelSpec(adapter_name='linear_regressor',
                      display_name='LinearRegressor'),
        ),
        metrics=('rmse', 'mae', 'r2'),
        artifact_spec=ArtifactSpec(
            output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(run_name='real_local_tser', primary_metric='rmse'),
    )

    result = run_tser_benchmark_suite(config)

    assert any(record.status is RunStatus.SUCCESS for record in result.run_records)
    assert any(record.model_name ==
               'LinearRegressor' for record in result.run_records)
