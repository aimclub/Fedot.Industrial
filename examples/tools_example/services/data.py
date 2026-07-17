from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from examples.tools_example.contracts import ToolRequest, ToolResponse
from examples.tools_example.services.common import PROJECT_ROOT, resolve_project_path

DATA_ROOT = PROJECT_ROOT / "examples" / "utils" / "data"


def list_local_datasets(request: ToolRequest) -> ToolResponse:
    task_type = str(request.payload.get("task_type", "")).strip()
    root = DATA_ROOT / task_type if task_type else DATA_ROOT
    datasets = []
    if root.exists():
        datasets = [
            {
                "name": path.name,
                "path": str(path),
                "is_dir": path.is_dir(),
            }
            for path in sorted(root.iterdir())
            if path.name not in {"__pycache__"} and not path.name.endswith(".py")
        ]
    return ToolResponse(
        name=request.name,
        status="dry_run" if request.dry_run else "success",
        dry_run=request.dry_run,
        message="Local dataset catalog resolved.",
        data={"root": str(root), "datasets": datasets},
    )


def load_dataset_preview(request: ToolRequest) -> ToolResponse:
    path = resolve_project_path(request.payload.get("path"))
    if path is None:
        return ToolResponse(
            name=request.name,
            status="failed",
            dry_run=request.dry_run,
            message="Dataset path is required.",
        )
    if request.dry_run:
        return ToolResponse(
            name=request.name,
            status="dry_run",
            dry_run=True,
            message="Dataset path resolved without loading data.",
            data={"path": str(path), "exists": path.exists()},
        )
    if not path.exists():
        return ToolResponse(
            name=request.name,
            status="failed",
            dry_run=False,
            message=f"Dataset path does not exist: {path}",
        )
    preview = _read_preview(path)
    return ToolResponse(
        name=request.name,
        status="success",
        dry_run=False,
        message="Dataset preview loaded.",
        data=preview,
    )


def _read_preview(path: Path) -> dict[str, Any]:
    if path.is_dir():
        files = [item for item in sorted(path.rglob("*")) if item.is_file()]
        return {
            "path": str(path),
            "file_count": len(files),
            "sample_files": [str(item) for item in files[:10]],
        }
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        separator = "\t" if suffix == ".tsv" else None
        frame = pd.read_csv(path, sep=separator, engine="python", nrows=10)
        return {
            "path": str(path),
            "shape_preview": [int(frame.shape[0]), int(frame.shape[1])],
            "columns": [str(column) for column in frame.columns],
            "records": frame.head(5).to_dict(orient="records"),
        }
    return {"path": str(path), "size_bytes": path.stat().st_size}


__all__ = ["list_local_datasets", "load_dataset_preview"]
