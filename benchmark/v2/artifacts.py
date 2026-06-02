from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .core import (
    ArtifactRecord,
    BenchmarkRunRecord,
    BenchmarkSuiteConfig,
    ensure_directory,
    to_plain_data,
    write_json,
)


def sanitize_artifact_payload(value: Any) -> Any:
    plain = to_plain_data(value)
    if isinstance(plain, float):
        return plain if math.isfinite(plain) else None
    if isinstance(plain, dict):
        return {str(key): sanitize_artifact_payload(item) for key, item in plain.items()}
    if isinstance(plain, list):
        return [sanitize_artifact_payload(item) for item in plain]
    if isinstance(plain, tuple):
        return [sanitize_artifact_payload(item) for item in plain]
    return plain


def slugify_artifact_part(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value).strip())
    normalized = normalized.strip("._")
    return normalized or "unnamed"


class IncrementalBenchmarkArtifactWriter:
    def __init__(self, config: BenchmarkSuiteConfig, run_id: str):
        self.enabled = bool(config.artifact_spec.persist_on_run)
        self.root_dir = Path(config.artifact_spec.output_dir) / run_id
        self.records_dir = self.root_dir / "records"
        self._manifest: dict[str, ArtifactRecord] = {}
        if self.enabled:
            ensure_directory(self.records_dir)

    def artifact_manifest(self) -> tuple[ArtifactRecord, ...]:
        return tuple(self._manifest.values())

    def write_run(
            self,
            run_record: BenchmarkRunRecord,
            *,
            metric_records: Sequence[Any] = (),
            prediction_records: Sequence[Any] = (),
            model_artifacts: dict[str, Any] | None = None,
    ) -> BenchmarkRunRecord:
        if not self.enabled:
            return run_record

        run_dir = ensure_directory(
            self.root_dir
            / "runs"
            / slugify_artifact_part(run_record.dataset_name)
            / slugify_artifact_part(run_record.model_name)
        )
        model_artifacts = model_artifacts or {}
        artifact_paths = {
            "run": str(run_dir / "run.json"),
            "metrics": str(run_dir / "metrics.json"),
            "predictions": str(run_dir / "predictions.csv"),
        }
        if model_artifacts.get("kernel_diagnostics") is not None:
            artifact_paths["kernel_diagnostics"] = str(run_dir / "kernel_diagnostics.json")
        if model_artifacts.get("kernel_selection") is not None:
            artifact_paths["kernel_selection"] = str(run_dir / "kernel_selection.json")
        if model_artifacts.get("model_diagnostics") is not None:
            artifact_paths["model_diagnostics"] = str(run_dir / "model_diagnostics.json")

        metadata = dict(run_record.metadata)
        metadata["artifact_paths"] = artifact_paths
        summary = model_artifacts.get("summary")
        if summary:
            if model_artifacts.get("kernel_diagnostics") is not None or model_artifacts.get(
                    "kernel_selection") is not None:
                metadata["kernel_learning_summary"] = sanitize_artifact_payload(summary)
            metadata["model_summary"] = sanitize_artifact_payload(summary)
        enriched_run_record = replace(run_record, metadata=metadata)

        self._write_per_run_files(
            run_dir,
            enriched_run_record,
            metric_records=metric_records,
            prediction_records=prediction_records,
            model_artifacts=model_artifacts,
        )
        self._append_records(enriched_run_record, metric_records, prediction_records, model_artifacts)
        return enriched_run_record

    def _write_per_run_files(
            self,
            run_dir: Path,
            run_record: BenchmarkRunRecord,
            *,
            metric_records: Sequence[Any],
            prediction_records: Sequence[Any],
            model_artifacts: dict[str, Any],
    ) -> None:
        run_path = run_dir / "run.json"
        metrics_path = run_dir / "metrics.json"
        predictions_path = run_dir / "predictions.csv"

        write_json(run_path, sanitize_artifact_payload(run_record))
        write_json(metrics_path, sanitize_artifact_payload(list(metric_records)))
        pd.DataFrame([sanitize_artifact_payload(record) for record in prediction_records]).to_csv(
            predictions_path,
            index=False,
        )
        self._remember("run", run_path, "json")
        self._remember("metrics", metrics_path, "json")
        self._remember("predictions", predictions_path, "csv")

        kernel_diagnostics = model_artifacts.get("kernel_diagnostics")
        if kernel_diagnostics is not None:
            path = run_dir / "kernel_diagnostics.json"
            write_json(path, sanitize_artifact_payload(kernel_diagnostics))
            self._remember("kernel_diagnostics", path, "json")

        kernel_selection = model_artifacts.get("kernel_selection")
        if kernel_selection is not None:
            path = run_dir / "kernel_selection.json"
            write_json(path, sanitize_artifact_payload(kernel_selection))
            self._remember("kernel_selection", path, "json")

        model_diagnostics = model_artifacts.get("model_diagnostics")
        if model_diagnostics is not None:
            path = run_dir / "model_diagnostics.json"
            write_json(path, sanitize_artifact_payload(model_diagnostics))
            self._remember("model_diagnostics", path, "json")

    def _append_records(
            self,
            run_record: BenchmarkRunRecord,
            metric_records: Sequence[Any],
            prediction_records: Sequence[Any],
            model_artifacts: dict[str, Any],
    ) -> None:
        self._append_jsonl(self.records_dir / "runs.jsonl", sanitize_artifact_payload(run_record))
        for record in metric_records:
            self._append_jsonl(self.records_dir / "metrics.jsonl", sanitize_artifact_payload(record))
        for record in prediction_records:
            self._append_jsonl(self.records_dir / "predictions.jsonl", sanitize_artifact_payload(record))

        if model_artifacts.get("kernel_diagnostics") is not None or model_artifacts.get("kernel_selection") is not None:
            payload = {
                "run_id": run_record.run_id,
                "benchmark": run_record.benchmark,
                "dataset_name": run_record.dataset_name,
                "subset": run_record.subset,
                "series_id": run_record.series_id,
                "model_name": run_record.model_name,
                "status": run_record.status,
                "kernel_diagnostics": model_artifacts.get("kernel_diagnostics"),
                "kernel_selection": model_artifacts.get("kernel_selection"),
                "summary": model_artifacts.get("summary"),
            }
            self._append_jsonl(self.records_dir / "kernel_diagnostics.jsonl", sanitize_artifact_payload(payload))
        if model_artifacts.get("kernel_selection") is not None:
            selection_payload = {
                "run_id": run_record.run_id,
                "benchmark": run_record.benchmark,
                "dataset_name": run_record.dataset_name,
                "subset": run_record.subset,
                "series_id": run_record.series_id,
                "model_name": run_record.model_name,
                "status": run_record.status,
                "kernel_selection": model_artifacts.get("kernel_selection"),
                "summary": model_artifacts.get("summary"),
            }
            self._append_jsonl(self.records_dir / "kernel_selection.jsonl",
                               sanitize_artifact_payload(selection_payload))
        if model_artifacts.get("model_diagnostics") is not None:
            diagnostics_payload = {
                "run_id": run_record.run_id,
                "benchmark": run_record.benchmark,
                "dataset_name": run_record.dataset_name,
                "subset": run_record.subset,
                "series_id": run_record.series_id,
                "model_name": run_record.model_name,
                "status": run_record.status,
                "model_diagnostics": model_artifacts.get("model_diagnostics"),
                "summary": model_artifacts.get("summary"),
            }
            self._append_jsonl(self.records_dir / "model_diagnostics.jsonl",
                               sanitize_artifact_payload(diagnostics_payload))

    def _append_jsonl(self, path: Path, payload: Any) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(sanitize_artifact_payload(payload), ensure_ascii=False, allow_nan=False) + "\n")
        self._remember("structured", path, "jsonl")

    def _remember(self, kind: str, path: Path, format_: str) -> None:
        key = str(path)
        self._manifest.setdefault(key, ArtifactRecord(kind=kind, path=key, format=format_))
