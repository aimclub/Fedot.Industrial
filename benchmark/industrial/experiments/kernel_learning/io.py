from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark.industrial.core import BenchmarkRunRecord


def read_json_if_exists(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    return json.loads(source.read_text(encoding="utf-8"))


def read_csv_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    return pd.read_csv(source).to_dict(orient="records")


def read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    records = []
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def status_counts(records: tuple[BenchmarkRunRecord, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.status.value] = counts.get(record.status.value, 0) + 1
    return counts


def load_stage1_kernel_records(stage1_output_dir: str | Path, run_id: str) -> dict[str, dict[str, Any]]:
    root = Path(stage1_output_dir) / run_id
    records_path = root / "records" / "kernel_selection.jsonl"
    if not records_path.exists():
        records_path = root / "records" / "kernel_diagnostics.jsonl"

    by_dataset: dict[str, dict[str, Any]] = {}
    for payload in read_jsonl_records(records_path):
        selection = payload.get("kernel_selection")
        if selection:
            by_dataset[payload["dataset_name"]] = payload
    if by_dataset:
        return by_dataset

    for selection_path in root.glob("runs/*/*/kernel_selection.json"):
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        dataset_name = selection_path.parents[1].name
        by_dataset[dataset_name] = {
            "dataset_name": dataset_name,
            "model_name": selection_path.parent.name,
            "kernel_selection": payload,
        }
    return by_dataset


_read_json_if_exists = read_json_if_exists
_read_csv_records = read_csv_records
_read_jsonl_records = read_jsonl_records
_status_counts = status_counts
