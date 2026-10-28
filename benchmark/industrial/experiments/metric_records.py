"""Persisted metric records shared by supervised benchmark tasks."""
import json
from pathlib import Path
from benchmark.industrial.core import MetricRecord, RunStatus


def load_metric_records(path: str | None) -> list[MetricRecord]:
    if not path or not Path(path).exists():
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = []
    for item in payload:
        records.append(
            MetricRecord(
                run_id=str(item["run_id"]),
                benchmark=str(item["benchmark"]),
                dataset_name=str(item["dataset_name"]),
                subset=str(item["subset"]),
                series_id=str(item["series_id"]),
                model_name=str(item["model_name"]),
                metric_name=str(item["metric_name"]),
                metric_value=float(item["metric_value"]),
                status=RunStatus(str(item["status"])),
            )
        )
    return records
