from __future__ import annotations

import json
from pathlib import Path

from .core import (
    ArtifactRecord,
    BenchmarkSuiteConfig,
    ClassificationBenchmarkResult,
    ForecastingBenchmarkResult,
    RegressionBenchmarkResult,
    TaskType,
    write_json,
)
from .evaluation.analytics import compare_models_on_series, render_publication_pack
from .evaluation.okhs_quality import render_okhs_smoothing_acceptance_pack


def _build_issue_artifacts(result, output_dir: str | Path) -> tuple[ArtifactRecord, ...]:
    target_dir = Path(output_dir)
    issues = []
    for record in result.run_records:
        if record.status.value == 'success':
            continue
        issues.append(
            {
                'run_id': record.run_id,
                'benchmark': record.benchmark,
                'dataset_name': record.dataset_name,
                'subset': record.subset,
                'series_id': record.series_id,
                'model_name': record.model_name,
                'status': record.status.value,
                'message': record.message,
                'tags': list(record.tags),
                'metadata': record.metadata,
            }
        )
    if not issues:
        return ()

    jsonl_path = target_dir / 'errors.jsonl'
    with jsonl_path.open('w', encoding='utf-8') as stream:
        for issue in issues:
            stream.write(json.dumps(issue, ensure_ascii=False) + '\n')

    summary_path = target_dir / 'errors_summary.json'
    status_counts: dict[str, int] = {}
    for issue in issues:
        status = str(issue['status'])
        status_counts[status] = status_counts.get(status, 0) + 1
    write_json(
        summary_path,
        {
            'run_id': result.run_id,
            'issue_count': len(issues),
            'status_counts': status_counts,
        },
    )
    return (
        ArtifactRecord(kind='structured', path=str(
            jsonl_path), format='jsonl'),
        ArtifactRecord(kind='structured', path=str(
            summary_path), format='json'),
    )


def run_forecasting_benchmark_suite(config: BenchmarkSuiteConfig) -> ForecastingBenchmarkResult:
    from .forecasting import run_forecasting_suite

    result = run_forecasting_suite(config)
    if config.artifact_spec.persist_on_run:
        output_dir = Path(config.artifact_spec.output_dir) / result.run_id
        manifest = list(result.artifact_manifest)
        manifest.extend(
            render_publication_pack(
                result,
                output_dir=output_dir,
            )
        )
        manifest.extend(render_okhs_smoothing_acceptance_pack(
            result, output_dir / 'aggregate'))
        manifest.extend(_build_issue_artifacts(result, output_dir))
        result = ForecastingBenchmarkResult(
            run_id=result.run_id,
            config=result.config,
            series_records=result.series_records,
            run_records=result.run_records,
            prediction_records=result.prediction_records,
            metric_records=result.metric_records,
            aggregate_report=result.aggregate_report,
            artifact_manifest=tuple(manifest),
        )
    return result


def compare_forecasting_models_on_series(
        result: ForecastingBenchmarkResult,
        series_id: str,
        output_dir: str | Path | None = None,
):
    return compare_models_on_series(result, series_id=series_id, output_dir=output_dir)


def build_forecasting_publication_pack(
        result: ForecastingBenchmarkResult,
        output_dir: str | Path | None = None,
) -> tuple[ArtifactRecord, ...]:
    return render_publication_pack(result, output_dir=output_dir)


def build_tsc_publication_pack(
        result: ClassificationBenchmarkResult,
        output_dir: str | Path | None = None,
) -> tuple[ArtifactRecord, ...]:
    from .classification import render_tsc_publication_pack

    return render_tsc_publication_pack(result, output_dir=output_dir)


def build_tser_publication_pack(
        result: RegressionBenchmarkResult,
        output_dir: str | Path | None = None,
) -> tuple[ArtifactRecord, ...]:
    from .regression import render_tser_publication_pack

    return render_tser_publication_pack(result, output_dir=output_dir)


def run_tsc_benchmark_suite(config: BenchmarkSuiteConfig):
    from .classification import render_tsc_publication_pack, run_tsc_suite

    if config.task_type is not TaskType.TS_CLASSIFICATION:
        raise ValueError(
            'run_tsc_benchmark_suite expects task_type=ts_classification.')
    result = run_tsc_suite(config)
    if config.artifact_spec.persist_on_run:
        output_dir = Path(config.artifact_spec.output_dir) / result.run_id
        manifest = list(result.artifact_manifest)
        manifest.extend(
            render_tsc_publication_pack(
                result,
                output_dir=output_dir,
            )
        )
        manifest.extend(_build_issue_artifacts(result, output_dir))
        result = ClassificationBenchmarkResult(
            run_id=result.run_id,
            config=result.config,
            dataset_records=result.dataset_records,
            run_records=result.run_records,
            prediction_records=result.prediction_records,
            metric_records=result.metric_records,
            aggregate_report=result.aggregate_report,
            artifact_manifest=tuple(manifest),
        )
    return result


def run_tser_benchmark_suite(config: BenchmarkSuiteConfig):
    from .regression import render_tser_publication_pack, run_tser_suite

    if config.task_type is not TaskType.TS_REGRESSION:
        raise ValueError(
            'run_tser_benchmark_suite expects task_type=ts_regression.')
    result = run_tser_suite(config)
    if config.artifact_spec.persist_on_run:
        output_dir = Path(config.artifact_spec.output_dir) / result.run_id
        manifest = list(result.artifact_manifest)
        manifest.extend(
            render_tser_publication_pack(
                result,
                output_dir=output_dir,
            )
        )
        manifest.extend(_build_issue_artifacts(result, output_dir))
        result = RegressionBenchmarkResult(
            run_id=result.run_id,
            config=result.config,
            dataset_records=result.dataset_records,
            run_records=result.run_records,
            prediction_records=result.prediction_records,
            metric_records=result.metric_records,
            aggregate_report=result.aggregate_report,
            artifact_manifest=tuple(manifest),
        )
    return result
