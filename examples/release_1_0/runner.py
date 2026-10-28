"""Explicit execution, TEST-only metrics and small inspectable artifact outputs."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np

from .config import parse_config
from .data import load_holdout, normalize_train_only


def run(config, output_dir, *, data=None):
    from fedot_ind.core.optimizer.evaluation import (
        EvaluationPlan, EvaluationTask, ExecutionStatus, LocalBackend, execute_plan,
    )
    from .models import fit_predict

    config = parse_config(config.to_record())
    data = load_holdout(config) if data is None else data
    train, test, normalization = normalize_train_only(
        data.train_x, data.test_x)
    plan = EvaluationPlan((EvaluationTask(config.scenario, config.seed),), LocalBackend(config.backend),
                          max_workers=1, candidate_timeout_seconds=120, total_timeout_seconds=150)
    result, = execute_plan(
        plan, ((config, train, data.train_y, test),), fit_predict)
    if result.status is not ExecutionStatus.SUCCEEDED:
        raise RuntimeError(
            f"{result.status.value}: {result.error.error_type}: {result.error.message}")
    details = dict(result.value)
    prediction = np.asarray(details.pop("prediction")).reshape(-1)
    actual = np.asarray(data.test_y).reshape(-1)
    if config.scenario == "airline":
        prediction = prediction * \
            normalization["scale"][0] + normalization["mean"][0]
    if prediction.shape != actual.shape or not np.isfinite(prediction).all():
        raise ValueError(
            "Expected one finite prediction for every TEST target")
    if config.scenario == "airline":
        metrics = {"mae": float(np.abs(prediction - actual).mean()),
                   "rmse": float(np.sqrt(np.square(prediction - actual).mean()))}
    else:
        from sklearn.metrics import accuracy_score, f1_score
        metrics = {"accuracy": float(accuracy_score(actual, prediction)),
                   "f1_macro": float(f1_score(actual, prediction, average="macro"))}
    digest = hashlib.sha256()
    for array in (data.train_x, data.train_y, data.test_x, data.test_y):
        if array is not None:
            array = np.ascontiguousarray(array)
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
    report = {"config": config.to_record(), "source": {**data.source, "content_sha256": digest.hexdigest()},
              "normalization": normalization, "train_count": len(train), "test_count": len(actual),
              "metrics": metrics, "status": result.status.value, **details,
              "versions": {name: version(name) for name in ("numpy", "torch", "sktime", "scikit-learn")},
              "python": platform.python_version()}
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "result.json").write_text(json.dumps(report,
                                                   indent=2, allow_nan=False) + "\n", encoding="utf-8")
    import csv
    with (target / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("test_id", "y_true", "y_pred"))
        writer.writerows(
            zip(data.test_ids, actual.tolist(), prediction.tolist()))
    if config.scenario == "airline":
        from benchmark.industrial.visualization import render_forecast_comparison_pack
        render_forecast_comparison_pack(
            history=data.train_x, actual=actual, forecasts={
                report["model"]: prediction},
            output_dir=target / "forecast_pack", title="Airline: fixed chronological holdout",
            series_id="Airline", source_metadata=report["source"])
    return report
