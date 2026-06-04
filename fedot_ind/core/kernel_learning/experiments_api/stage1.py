from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from benchmark.industrial.classification import render_tsc_publication_pack, run_tsc_suite
from benchmark.industrial.core import (
    ArtifactSpec,
    BenchmarkAggregateReport,
    BenchmarkRunRecord,
    BenchmarkSuiteConfig,
    ClassificationBenchmarkResult,
    DatasetSpec,
    LabelPredictionRecord,
    MetricRecord,
    ModelSpec,
    RunSpec,
    RunStatus,
    TaskType,
)
from benchmark.industrial.datasets.discovery import discover_local_ucr_datasets
from fedot_ind.core.repository.constanst_repository import UNI_CLF_BENCH
from .io import load_stage1_kernel_records, read_csv_records, read_json_if_exists, status_counts

DEFAULT_STAGE1_GENERATORS = (
    "quantile_extractor_torch",
    "wavelet_extractor",
    "fourier_extractor",
    "recurrence_extractor",
)
DEFAULT_STAGE_METRICS = ("accuracy", "balanced_accuracy", "f1_macro")


@dataclass(frozen=True)
class KernelLearningStage1Runner:
    data_root: str | Path
    output_dir: str | Path
    datasets: tuple[str, ...] = ()
    allowed_dataset_names: Sequence[str] = field(default_factory=lambda: tuple(UNI_CLF_BENCH))
    generator_names: tuple[str, ...] = DEFAULT_STAGE1_GENERATORS
    metrics: tuple[str, ...] = DEFAULT_STAGE_METRICS
    run_name: str = "kernel_learning_ucr_stage1"
    model_display_name: str = "KernelEnsembleClassifier_all_non_topological"
    importance_threshold: float = 0.15
    show_progress: bool = True

    def resolve_ucr_datasets(self) -> tuple[str, ...]:
        if self.datasets:
            return tuple(self.datasets)
        return discover_local_ucr_datasets(self.data_root, allowed_names=self.allowed_dataset_names)

    def build_config(self) -> BenchmarkSuiteConfig:
        return BenchmarkSuiteConfig(
            task_type=TaskType.TS_CLASSIFICATION,
            datasets=tuple(
                DatasetSpec(
                    benchmark="ucr",
                    dataset_name=dataset_name,
                    adapter_options={
                        "local_data_root": str(self.data_root),
                        "download_if_missing": True,
                    },
                )
                for dataset_name in self.resolve_ucr_datasets()
            ),
            models=(
                ModelSpec(
                    adapter_name="kernel_ensemble_classifier",
                    display_name=self.model_display_name,
                    params={
                        "generator_names": self.generator_names,
                        "kernel": "rbf",
                        "gamma": "scale",
                        "torch_device": "auto",
                        "C": 1.0,
                        "complexity_penalty": 0.01,
                        "redundancy_penalty": 0.05,
                        "min_weight": 0.05,
                        "importance_threshold": self.importance_threshold,
                    },
                ),
            ),
            metrics=self.metrics,
            artifact_spec=ArtifactSpec(output_dir=str(self.output_dir), persist_on_run=True),
            run_spec=RunSpec(
                run_name=self.run_name,
                primary_metric="f1_macro",
                show_progress=self.show_progress,
                progress_leave=False,
                progress_log_errors=True,
                progress_log_summaries=True,
            ),
        )

    def run(self) -> ClassificationBenchmarkResult:
        config = self.build_config()
        result = run_tsc_suite(config)
        if config.artifact_spec.persist_on_run:
            output_dir = Path(config.artifact_spec.output_dir) / result.run_id
            manifest = list(result.artifact_manifest)
            manifest.extend(render_tsc_publication_pack(result, output_dir=output_dir))
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


@dataclass(frozen=True)
class KernelLearningStage1ArtifactsLoader:
    data_root: str | Path
    fallback_generators: tuple[str, ...] = DEFAULT_STAGE1_GENERATORS
    fallback_metrics: tuple[str, ...] = DEFAULT_STAGE_METRICS
    fallback_model_display_name: str = "KernelEnsembleClassifier_all_non_topological"

    def load(self, run_dir: str | Path, *, load_predictions: bool = False) -> ClassificationBenchmarkResult:
        root = Path(run_dir)
        aggregate_dir = root / "aggregate"
        metadata = read_json_if_exists(aggregate_dir / "run_metadata.json")
        dataset_specs = self._load_dataset_specs_from_metadata(metadata, root)
        model_specs = self._load_model_specs_from_metadata(metadata, root)
        run_records = self._load_run_records(root)
        metric_records = self._load_metric_records(root)
        prediction_records = self._load_prediction_records(root) if load_predictions else ()
        primary_metric = "f1_macro"

        return ClassificationBenchmarkResult(
            run_id=root.name,
            config=BenchmarkSuiteConfig(
                task_type=TaskType.TS_CLASSIFICATION,
                datasets=dataset_specs,
                models=model_specs,
                metrics=tuple(sorted({record.metric_name for record in metric_records})) or self.fallback_metrics,
                artifact_spec=ArtifactSpec(output_dir=str(root.parent), persist_on_run=True),
                run_spec=RunSpec(run_name="kernel_learning_ucr_stage1", primary_metric=primary_metric),
            ),
            dataset_records=(),
            run_records=run_records,
            prediction_records=prediction_records,
            metric_records=metric_records,
            aggregate_report=BenchmarkAggregateReport(
                run_id=root.name,
                task_type=TaskType.TS_CLASSIFICATION,
                primary_metric=primary_metric,
                leaderboard_rows=tuple(read_csv_records(aggregate_dir / "leaderboard.csv")),
                status_counts=status_counts(run_records),
            ),
        )

    def _load_dataset_specs_from_metadata(self, metadata: dict[str, Any], run_dir: Path) -> tuple[DatasetSpec, ...]:
        payloads = tuple(metadata.get("dataset_specs") or ())
        if payloads:
            return tuple(DatasetSpec(**payload) for payload in payloads)
        kernel_records = load_stage1_kernel_records(run_dir.parent, run_dir.name)
        return tuple(
            DatasetSpec(
                benchmark="ucr",
                dataset_name=dataset_name,
                adapter_options={
                    "local_data_root": str(self.data_root),
                    "download_if_missing": True,
                },
            )
            for dataset_name in sorted(kernel_records)
        )

    def _load_model_specs_from_metadata(self, metadata: dict[str, Any], run_dir: Path) -> tuple[ModelSpec, ...]:
        payloads = tuple(metadata.get("model_specs") or ())
        if payloads:
            return tuple(ModelSpec(**payload) for payload in payloads)
        kernel_records = load_stage1_kernel_records(run_dir.parent, run_dir.name)
        model_names = tuple(
            sorted(
                {
                    str(record.get("model_name"))
                    for record in kernel_records.values()
                    if record.get("model_name")
                }
            )
        )
        if model_names:
            return tuple(
                ModelSpec(
                    adapter_name="kernel_ensemble_classifier",
                    display_name=model_name,
                    params={"generator_names": self.fallback_generators},
                )
                for model_name in model_names
            )
        return (
            ModelSpec(
                adapter_name="kernel_ensemble_classifier",
                display_name=self.fallback_model_display_name,
                params={"generator_names": self.fallback_generators},
            ),
        )

    def _load_run_records(self, run_dir: Path) -> tuple[BenchmarkRunRecord, ...]:
        records = []
        for row in read_csv_records(run_dir / "aggregate" / "runs.csv"):
            metric_summary = {
                key: float(value)
                for key, value in row.items()
                if key not in {"run_id", "benchmark", "dataset_name", "subset", "model_name", "status"}
                and pd.notna(value)
            }
            dataset_name = str(row.get("dataset_name", ""))
            records.append(
                BenchmarkRunRecord(
                    run_id=str(row.get("run_id") or run_dir.name),
                    benchmark=str(row.get("benchmark") or "ucr_uea"),
                    dataset_name=dataset_name,
                    subset=str(row.get("subset") or "default"),
                    series_id=dataset_name,
                    model_name=str(row.get("model_name") or self.fallback_model_display_name),
                    status=RunStatus(str(row.get("status") or "success")),
                    metrics_summary=metric_summary,
                )
            )
        return tuple(records)

    def _load_metric_records(self, run_dir: Path) -> tuple[MetricRecord, ...]:
        records = []
        for row in read_csv_records(run_dir / "aggregate" / "metrics.csv"):
            records.append(
                MetricRecord(
                    run_id=str(row.get("run_id") or run_dir.name),
                    benchmark=str(row.get("benchmark") or "ucr_uea"),
                    dataset_name=str(row.get("dataset_name") or ""),
                    subset=str(row.get("subset") or "default"),
                    series_id=str(row.get("series_id") or row.get("dataset_name") or ""),
                    model_name=str(row.get("model_name") or self.fallback_model_display_name),
                    metric_name=str(row.get("metric_name") or ""),
                    metric_value=float(row.get("metric_value")),
                    status=RunStatus(str(row.get("status") or "success")),
                )
            )
        return tuple(records)

    def _load_prediction_records(self, run_dir: Path) -> tuple[LabelPredictionRecord, ...]:
        records = []
        for row in read_csv_records(run_dir / "aggregate" / "predictions.csv"):
            records.append(
                LabelPredictionRecord(
                    run_id=str(row.get("run_id") or run_dir.name),
                    benchmark=str(row.get("benchmark") or "ucr_uea"),
                    dataset_name=str(row.get("dataset_name") or ""),
                    subset=str(row.get("subset") or "default"),
                    model_name=str(row.get("model_name") or self.fallback_model_display_name),
                    sample_index=int(row.get("sample_index")),
                    y_true=str(row.get("y_true")),
                    y_pred=str(row.get("y_pred")),
                    status=RunStatus(str(row.get("status") or "success")),
                )
            )
        return tuple(records)


def resolve_existing_stage1_run_dir(
        *,
        stage1_output_dir: str | Path,
        run_id: str | None = None,
) -> Path:
    output_dir = Path(stage1_output_dir)
    if run_id:
        candidate = output_dir / run_id
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Stage 1 run directory was not found: {candidate}")
    candidates = sorted(
        (path for path in output_dir.glob("kernel_learning_ucr_stage1_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No existing stage 1 runs found in {output_dir}")
    return candidates[0]


def load_stage1_result_from_artifacts(
        run_dir: str | Path,
        *,
        data_root: str | Path,
        load_predictions: bool = False,
        fallback_generators: tuple[str, ...] = DEFAULT_STAGE1_GENERATORS,
        fallback_metrics: tuple[str, ...] = DEFAULT_STAGE_METRICS,
) -> ClassificationBenchmarkResult:
    return KernelLearningStage1ArtifactsLoader(
        data_root=data_root,
        fallback_generators=fallback_generators,
        fallback_metrics=fallback_metrics,
    ).load(run_dir, load_predictions=load_predictions)
