from __future__ import annotations

from pathlib import Path

import pandas as pd

from examples.tools_example.contracts import ToolRequest, ToolResponse
from examples.tools_example.services.common import PROJECT_ROOT, load_tool_defaults


def anomaly_detection_context() -> dict:
    defaults = load_tool_defaults()
    payload = dict(defaults["anomaly_detection"])
    payload["data_root"] = str(PROJECT_ROOT / payload["data_root"])
    return payload


def detect_anomalies(request: ToolRequest) -> ToolResponse:
    context = anomaly_detection_context()
    folder = str(request.payload.get("folder", "valve1"))
    dataset = str(request.payload.get("dataset", "0"))
    data_root = Path(request.payload.get("data_root") or context["data_root"])
    target_path = data_root / folder / f"{dataset}.csv"
    if request.dry_run:
        return ToolResponse(
            name=request.name,
            status="dry_run",
            dry_run=True,
            message="Anomaly detection context resolved; execution was not requested.",
            data={**context, "target_path": str(target_path), "exists": target_path.exists()},
        )
    if not target_path.exists():
        return ToolResponse(
            name=request.name,
            status="failed",
            dry_run=False,
            message=f"Anomaly dataset does not exist: {target_path}",
            data={**context, "target_path": str(target_path), "exists": False},
        )

    frame = pd.read_csv(target_path, sep=";", engine="python")
    numeric = frame.select_dtypes(include="number")
    scores = numeric.std(axis=1).fillna(0.0) if not numeric.empty else pd.Series([0.0] * len(frame))
    threshold = float(scores.quantile(float(request.payload.get("threshold_quantile", 0.99))))
    labels = (scores >= threshold).astype(int)
    return ToolResponse(
        name=request.name,
        status="success",
        dry_run=False,
        message="Lightweight anomaly scoring completed.",
        data={
            **context,
            "target_path": str(target_path),
            "rows": int(len(frame)),
            "threshold": threshold,
            "anomaly_count": int(labels.sum()),
            "scoring": "numeric_row_std_quantile_baseline",
        },
    )


__all__ = ["anomaly_detection_context", "detect_anomalies"]
