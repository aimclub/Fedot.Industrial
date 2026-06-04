from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class KernelStage1Analysis:
    run_dir: Path
    output_dir: Path | None
    summary: dict[str, Any]
    dataset_summary: pd.DataFrame
    generator_summary: pd.DataFrame
    recommendation_summary: pd.DataFrame
    kernel_diagnostics: pd.DataFrame
    artifact_manifest: tuple[dict[str, str], ...] = ()


class KernelStage1AnalysisError(ValueError):
    pass


def load_kernel_stage1_records(run_dir: str | Path) -> tuple[dict[str, Any], ...]:
    root = Path(run_dir)
    records_dir = root / "records"
    diagnostics_path = records_dir / "kernel_diagnostics.jsonl"
    selection_path = records_dir / "kernel_selection.jsonl"

    if diagnostics_path.exists():
        records = _read_jsonl(diagnostics_path)
    elif selection_path.exists():
        records = _read_jsonl(selection_path)
    else:
        records = _read_records_from_run_tree(root)

    if not records:
        raise KernelStage1AnalysisError(f"No kernel stage1 records found in {root}")
    return tuple(sorted(records, key=lambda item: (str(item.get("dataset_name", "")), str(item.get("model_name", "")))))


def analyze_kernel_stage1_run(
        run_dir: str | Path,
        *,
        output_dir: str | Path | None = None,
) -> KernelStage1Analysis:
    root = Path(run_dir)
    records = load_kernel_stage1_records(root)
    run_metrics = _load_run_metrics(root)
    kernel_rows = _build_kernel_rows(records)
    kernel_diagnostics = pd.DataFrame(kernel_rows)
    dataset_summary = _build_dataset_summary(records, kernel_diagnostics, run_metrics)
    generator_summary = _build_generator_summary(kernel_diagnostics)
    recommendation_summary = _build_recommendation_summary(dataset_summary)
    summary = _build_summary(root, records, dataset_summary, generator_summary, recommendation_summary,
                             kernel_diagnostics)
    return KernelStage1Analysis(
        run_dir=root,
        output_dir=Path(output_dir) if output_dir is not None else None,
        summary=summary,
        dataset_summary=dataset_summary,
        generator_summary=generator_summary,
        recommendation_summary=recommendation_summary,
        kernel_diagnostics=kernel_diagnostics,
    )


def render_kernel_stage1_summary_report(
        run_dir: str | Path,
        *,
        output_dir: str | Path | None = None,
        top_n: int = 12,
) -> KernelStage1Analysis:
    target_dir = _ensure_directory(output_dir or Path(run_dir) / "analysis")
    analysis = analyze_kernel_stage1_run(run_dir, output_dir=target_dir)

    artifacts: list[dict[str, str]] = []
    artifacts.extend(_write_frame_bundle(analysis.dataset_summary, target_dir / "dataset_kernel_summary"))
    artifacts.extend(_write_frame_bundle(analysis.generator_summary, target_dir / "generator_importance_summary"))
    artifacts.extend(_write_frame_bundle(analysis.recommendation_summary, target_dir / "recommendation_summary"))
    artifacts.extend(_write_frame_bundle(analysis.kernel_diagnostics, target_dir / "kernel_diagnostics_summary"))
    artifacts.extend(render_kernel_stage1_visualizations(analysis, target_dir))

    summary = {
        **analysis.summary,
        "visualizations": [artifact for artifact in artifacts if artifact["kind"] == "plot"],
    }
    analysis = KernelStage1Analysis(
        run_dir=analysis.run_dir,
        output_dir=target_dir,
        summary=summary,
        dataset_summary=analysis.dataset_summary,
        generator_summary=analysis.generator_summary,
        recommendation_summary=analysis.recommendation_summary,
        kernel_diagnostics=analysis.kernel_diagnostics,
    )

    summary_path = target_dir / "summary.json"
    _write_json(summary_path, analysis.summary)
    artifacts.append({"kind": "structured", "path": str(summary_path), "format": "json"})

    report_path = target_dir / "summary_report.md"
    report_path.write_text(_render_markdown_report(analysis, top_n=top_n), encoding="utf-8")
    artifacts.append({"kind": "summary", "path": str(report_path), "format": "md"})

    manifest_path = target_dir / "artifact_manifest.json"
    _write_json(manifest_path, artifacts)
    artifacts.append({"kind": "structured", "path": str(manifest_path), "format": "json"})

    return KernelStage1Analysis(
        run_dir=analysis.run_dir,
        output_dir=target_dir,
        summary=analysis.summary,
        dataset_summary=analysis.dataset_summary,
        generator_summary=analysis.generator_summary,
        recommendation_summary=analysis.recommendation_summary,
        kernel_diagnostics=analysis.kernel_diagnostics,
        artifact_manifest=tuple(artifacts),
    )


def render_kernel_stage1_visualizations(
        analysis: KernelStage1Analysis,
        output_dir: str | Path,
) -> list[dict[str, str]]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    target_dir = _ensure_directory(Path(output_dir) / "plots")
    artifacts: list[dict[str, str]] = []

    metric_columns = [
        column
        for column in ("accuracy", "balanced_accuracy", "f1_macro")
        if column in analysis.dataset_summary.columns
    ]
    if metric_columns:
        figure, axis = plt.subplots(figsize=(8, 5))
        values = [
            pd.to_numeric(analysis.dataset_summary[column], errors="coerce").dropna().to_numpy()
            for column in metric_columns
        ]
        axis.boxplot(values, labels=metric_columns, showmeans=True)
        axis.set_title("Stage 1 Metric Distribution")
        axis.set_ylabel("Metric value")
        axis.grid(axis="y", alpha=0.25)
        artifacts.extend(_save_figure_bundle(figure, target_dir / "metric_distribution_boxplot"))
        plt.close(figure)

    if not analysis.generator_summary.empty:
        frame = analysis.generator_summary.sort_values("important_count", ascending=False)
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.bar(frame["generator_name"], frame["important_count"], label="important")
        axis.bar(frame["generator_name"], frame["top1_count"], label="top-1")
        axis.set_title("Generator Importance Counts")
        axis.set_ylabel("Dataset count")
        axis.tick_params(axis="x", rotation=20)
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        artifacts.extend(_save_figure_bundle(figure, target_dir / "generator_importance_counts"))
        plt.close(figure)

    if not analysis.kernel_diagnostics.empty:
        generator_names = list(analysis.kernel_diagnostics["generator_name"].dropna().unique())
        if generator_names:
            weight_values = [
                pd.to_numeric(
                    analysis.kernel_diagnostics.loc[
                        analysis.kernel_diagnostics["generator_name"] == name,
                        "weight",
                    ],
                    errors="coerce",
                ).dropna().to_numpy()
                for name in generator_names
            ]
            figure, axis = plt.subplots(figsize=(9, 5))
            axis.boxplot(weight_values, labels=generator_names, showmeans=True)
            axis.set_title("Kernel Weight Distribution By Generator")
            axis.set_ylabel("Sparse MKL weight")
            axis.tick_params(axis="x", rotation=20)
            axis.grid(axis="y", alpha=0.25)
            artifacts.extend(_save_figure_bundle(figure, target_dir / "kernel_weight_by_generator_boxplot"))
            plt.close(figure)

            condition_values = []
            condition_labels = []
            for name in generator_names:
                series = pd.to_numeric(
                    analysis.kernel_diagnostics.loc[
                        analysis.kernel_diagnostics["generator_name"] == name,
                        "condition_number",
                    ],
                    errors="coerce",
                ).dropna()
                series = series[series > 0.0].map(math.log10)
                if not series.empty:
                    condition_labels.append(name)
                    condition_values.append(series.to_numpy())
            if condition_values:
                figure, axis = plt.subplots(figsize=(9, 5))
                axis.boxplot(condition_values, labels=condition_labels, showmeans=True)
                axis.set_title("Kernel Condition Number By Generator")
                axis.set_ylabel("log10(condition number)")
                axis.tick_params(axis="x", rotation=20)
                axis.grid(axis="y", alpha=0.25)
                artifacts.extend(_save_figure_bundle(figure, target_dir / "condition_number_by_generator_boxplot"))
                plt.close(figure)

    return artifacts


def _build_kernel_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        selection = dict(record.get("kernel_selection") or {})
        report = dict(selection.get("selection_report") or {})
        generator_names = list(report.get("generator_names") or selection.get("selected_generators") or ())
        weights = list(report.get("weights") or selection.get("selected_weights") or ())
        diagnostics_by_name = {
            item.get("name"): item
            for item in (dict(record.get("kernel_diagnostics") or {}).get("kernels") or ())
        }
        importance_rank = {
            item.get("name"): int(item.get("rank", index + 1))
            for index, item in enumerate(dict(selection.get("kernel_importance") or {}).get("items") or ())
        }
        important_names = set(selection.get("important_generators") or ())
        scores = dict(report.get("scores") or {})
        alignments = dict(report.get("alignments") or {})
        complexities = dict(report.get("complexities") or {})
        redundancies = dict(report.get("redundancies") or {})

        for index, generator_name in enumerate(generator_names):
            kernel_payload = dict(diagnostics_by_name.get(generator_name) or {})
            diagnostics = dict(kernel_payload.get("diagnostics") or {})
            feature_diag = dict(diagnostics.get("feature_generator") or {})
            train_shape = _as_shape(kernel_payload.get("train_kernel_shape"))
            test_shape = _as_shape(kernel_payload.get("test_kernel_shape"))
            rows.append(
                {
                    "run_id": record.get("run_id"),
                    "dataset_name": record.get("dataset_name"),
                    "model_name": record.get("model_name"),
                    "status": record.get("status"),
                    "generator_name": generator_name,
                    "weight": _float_at(weights, index),
                    "is_important": generator_name in important_names or generator_name in importance_rank,
                    "importance_rank": importance_rank.get(generator_name),
                    "score": _safe_float(scores.get(generator_name)),
                    "alignment": _safe_float(alignments.get(generator_name)),
                    "selector_complexity": _safe_float(complexities.get(generator_name)),
                    "redundancy": _safe_float(redundancies.get(generator_name)),
                    "kernel_complexity": _safe_float(dict(kernel_payload.get("complexity") or {}).get(
                        "kernel_complexity")),
                    "min_eigenvalue": _safe_float(diagnostics.get("min_eigenvalue")),
                    "condition_number": _safe_float(diagnostics.get("condition_number")),
                    "is_psd": bool(kernel_payload.get("is_psd", diagnostics.get("is_psd", True))),
                    "psd_correction": kernel_payload.get("psd_correction", diagnostics.get("psd_correction")),
                    "gamma": _safe_float(diagnostics.get("gamma")),
                    "kernel": diagnostics.get("kernel"),
                    "normalize": diagnostics.get("normalize"),
                    "n_features": _safe_int(feature_diag.get("n_features")),
                    "torch_device": feature_diag.get("torch_device", ""),
                    "operations": " -> ".join(str(item) for item in feature_diag.get("operations") or ()),
                    "train_kernel_rows": train_shape[0] if train_shape else None,
                    "train_kernel_cols": train_shape[1] if len(train_shape) > 1 else None,
                    "test_kernel_rows": test_shape[0] if test_shape else None,
                    "test_kernel_cols": test_shape[1] if len(test_shape) > 1 else None,
                }
            )
    return rows


def _build_dataset_summary(
        records: Iterable[dict[str, Any]],
        kernel_diagnostics: pd.DataFrame,
        run_metrics: dict[tuple[str, str], dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for record in records:
        selection = dict(record.get("kernel_selection") or {})
        important_generators = list(selection.get("important_generators") or ())
        important_weights = [_safe_float(value) for value in selection.get("important_weights") or ()]
        selected_generators = list(selection.get("selected_generators") or ())
        selected_weights = [_safe_float(value) for value in selection.get("selected_weights") or ()]
        dataset_name = str(record.get("dataset_name", ""))
        model_name = str(record.get("model_name", ""))
        dataset_kernel_rows = _filter_frame(kernel_diagnostics, dataset_name=dataset_name, model_name=model_name)
        metric_payload = run_metrics.get((dataset_name, model_name), {})
        top_generator = important_generators[0] if important_generators else _top_weighted_name(selected_generators,
                                                                                                selected_weights)
        rows.append(
            {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "status": record.get("status"),
                **metric_payload,
                "top_generator": top_generator,
                "top_weight": _safe_float(important_weights[0] if important_weights else None),
                "important_generators": " + ".join(important_generators),
                "important_weights": ", ".join(_format_float(weight) for weight in important_weights),
                "selected_generators": " + ".join(selected_generators),
                "selected_weights": ", ".join(_format_float(weight) for weight in selected_weights),
                "n_important_generators": len(important_generators),
                "n_selected_generators": len(selected_generators),
                "n_kernels": int(len(dataset_kernel_rows)),
                "psd_failures": int(
                    (dataset_kernel_rows["is_psd"] == False).sum()) if not dataset_kernel_rows.empty else 0,
                "min_eigenvalue_min": _frame_min(dataset_kernel_rows, "min_eigenvalue"),
                "condition_number_max": _frame_max(dataset_kernel_rows, "condition_number"),
                "torch_devices": ", ".join(sorted(set(str(value) for value in dataset_kernel_rows.get(
                    "torch_device", pd.Series(dtype=object)).dropna() if str(value)))),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        sort_columns = [column for column in ("f1_macro", "accuracy", "dataset_name") if column in frame.columns]
        ascending = [False if column in {"f1_macro", "accuracy", "balanced_accuracy"} else True
                     for column in sort_columns]
        frame = frame.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
    return frame


def _build_generator_summary(kernel_diagnostics: pd.DataFrame) -> pd.DataFrame:
    if kernel_diagnostics.empty:
        return pd.DataFrame()
    rows = []
    total_datasets = max(1, int(kernel_diagnostics["dataset_name"].nunique()))
    for generator_name, group in kernel_diagnostics.groupby("generator_name", sort=True):
        rows.append(
            {
                "generator_name": generator_name,
                "appeared_runs": int(group["dataset_name"].nunique()),
                "important_count": int(group["is_important"].sum()),
                "important_rate": float(group["is_important"].sum() / total_datasets),
                "top1_count": int((group["importance_rank"] == 1).sum()),
                "mean_weight": _series_mean(group["weight"]),
                "median_weight": _series_median(group["weight"]),
                "max_weight": _series_max(group["weight"]),
                "mean_alignment": _series_mean(group["alignment"]),
                "mean_score": _series_mean(group["score"]),
                "mean_selector_complexity": _series_mean(group["selector_complexity"]),
                "mean_redundancy": _series_mean(group["redundancy"]),
                "psd_failure_count": int((group["is_psd"] == False).sum()),
                "max_condition_number": _series_max(group["condition_number"]),
                "median_n_features": _series_median(group["n_features"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["top1_count", "important_count", "mean_weight", "generator_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def _build_recommendation_summary(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    if dataset_summary.empty or "important_generators" not in dataset_summary.columns:
        return pd.DataFrame()
    rows = []
    for generators, group in dataset_summary.groupby("important_generators", sort=True):
        rows.append(
            {
                "important_generators": generators,
                "dataset_count": int(len(group)),
                "datasets": ", ".join(sorted(group["dataset_name"].astype(str))),
                "mean_f1_macro": _series_mean(group["f1_macro"]) if "f1_macro" in group.columns else None,
                "mean_accuracy": _series_mean(group["accuracy"]) if "accuracy" in group.columns else None,
                "mean_n_important_generators": _series_mean(group["n_important_generators"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["dataset_count", "mean_f1_macro", "important_generators"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _build_summary(
        run_dir: Path,
        records: Iterable[dict[str, Any]],
        dataset_summary: pd.DataFrame,
        generator_summary: pd.DataFrame,
        recommendation_summary: pd.DataFrame,
        kernel_diagnostics: pd.DataFrame,
) -> dict[str, Any]:
    records = tuple(records)
    status_counts = dataset_summary["status"].value_counts().to_dict() if "status" in dataset_summary else {}
    metric_columns = [column for column in ("accuracy", "balanced_accuracy", "f1_macro") if column in dataset_summary]
    metric_summary = {
        metric: {
            "mean": _series_mean(dataset_summary[metric]),
            "median": _series_median(dataset_summary[metric]),
            "min": _series_min(dataset_summary[metric]),
            "max": _series_max(dataset_summary[metric]),
        }
        for metric in metric_columns
    }
    return {
        "run_id": records[0].get("run_id") if records else run_dir.name,
        "run_dir": str(run_dir),
        "dataset_count": int(dataset_summary["dataset_name"].nunique()) if "dataset_name" in dataset_summary else 0,
        "kernel_record_count": int(len(records)),
        "status_counts": status_counts,
        "metric_summary": metric_summary,
        "generator_count": int(generator_summary["generator_name"].nunique()) if not generator_summary.empty else 0,
        "top_generators": generator_summary.head(5).to_dict(orient="records") if not generator_summary.empty else [],
        "recommendation_count": int(len(recommendation_summary)),
        "psd_failure_count": int((kernel_diagnostics["is_psd"] == False).sum()) if not kernel_diagnostics.empty else 0,
        "max_condition_number": _series_max(
            kernel_diagnostics["condition_number"]) if not kernel_diagnostics.empty else None,
    }


def _render_markdown_report(analysis: KernelStage1Analysis, *, top_n: int) -> str:
    summary = analysis.summary
    dataset_summary = analysis.dataset_summary
    generator_summary = analysis.generator_summary
    recommendation_summary = analysis.recommendation_summary
    diagnostics = analysis.kernel_diagnostics

    lines = [
        "# Kernel Learning Stage 1 Summary",
        "",
        f"- Run ID: `{summary.get('run_id')}`",
        f"- Datasets: `{summary.get('dataset_count', 0)}`",
        f"- Kernel records: `{summary.get('kernel_record_count', 0)}`",
        f"- Status counts: `{summary.get('status_counts', {})}`",
        f"- PSD failures: `{summary.get('psd_failure_count', 0)}`",
        "",
        "## Metric Overview",
        "",
        _render_metric_summary(summary.get("metric_summary") or {}),
    ]

    if not dataset_summary.empty:
        metric = "f1_macro" if "f1_macro" in dataset_summary.columns else (
            "accuracy" if "accuracy" in dataset_summary.columns else None)
        if metric is not None:
            top_frame = dataset_summary.sort_values(metric, ascending=False).head(top_n)
            bottom_frame = dataset_summary.sort_values(metric, ascending=True).head(top_n)
            lines.extend(
                ["", f"## Top Datasets By {metric}", "", _frame_to_markdown(_compact_dataset_frame(top_frame))])
            lines.extend(
                ["", f"## Hardest Datasets By {metric}", "", _frame_to_markdown(_compact_dataset_frame(bottom_frame))])

    if not generator_summary.empty:
        lines.extend(
            [
                "",
                "## Generator Importance",
                "",
                _frame_to_markdown(generator_summary.head(top_n)),
            ]
        )

    if not recommendation_summary.empty:
        lines.extend(
            [
                "",
                "## Warm-Start Generator Sets",
                "",
                _frame_to_markdown(recommendation_summary.head(top_n)),
            ]
        )

    if not diagnostics.empty:
        watchlist = diagnostics.sort_values("condition_number", ascending=False).head(top_n)
        lines.extend(
            [
                "",
                "## Kernel Diagnostics Watchlist",
                "",
                _frame_to_markdown(
                    watchlist[
                        [
                            "dataset_name",
                            "generator_name",
                            "weight",
                            "condition_number",
                            "min_eigenvalue",
                            "is_psd",
                            "n_features",
                            "torch_device",
                        ]
                    ]
                ),
            ]
        )

    visualizations = analysis.summary.get("visualizations") or ()
    if visualizations:
        lines.extend(["", "## Visualizations", ""])
        for artifact in visualizations:
            path = Path(str(artifact["path"]))
            label = path.stem.replace("_", " ").title()
            rel_path = path.relative_to(analysis.output_dir) if analysis.output_dir and path.is_relative_to(
                analysis.output_dir) else path
            lines.append(f"- [{label}]({rel_path.as_posix()})")

    lines.extend(
        [
            "",
            "## Output Tables",
            "",
            "- `dataset_kernel_summary.csv`",
            "- `generator_importance_summary.csv`",
            "- `recommendation_summary.csv`",
            "- `kernel_diagnostics_summary.csv`",
            "- `summary.json`",
        ]
    )
    return "\n".join(lines)


def _load_run_metrics(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = run_dir / "aggregate" / "runs.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    metric_columns = [
        column
        for column in frame.columns
        if column not in {"run_id", "benchmark", "dataset_name", "subset", "model_name", "status"}
    ]
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        result[(str(row.get("dataset_name")), str(row.get("model_name")))] = {
            column: _safe_float(row.get(column))
            for column in metric_columns
        }
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_records_from_run_tree(root: Path) -> list[dict[str, Any]]:
    rows = []
    for selection_path in sorted((root / "runs").glob("*/*/kernel_selection.json")):
        dataset_name = selection_path.parents[1].name
        model_name = selection_path.parent.name
        payload = {
            "run_id": root.name,
            "dataset_name": dataset_name,
            "model_name": model_name,
            "status": "success",
            "kernel_selection": json.loads(selection_path.read_text(encoding="utf-8")),
        }
        diagnostics_path = selection_path.with_name("kernel_diagnostics.json")
        if diagnostics_path.exists():
            payload["kernel_diagnostics"] = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        rows.append(payload)
    return rows


def _write_frame_bundle(frame: pd.DataFrame, path_without_suffix: Path) -> list[dict[str, str]]:
    csv_path = path_without_suffix.with_suffix(".csv")
    json_path = path_without_suffix.with_suffix(".json")
    frame.to_csv(csv_path, index=False)
    json_path.write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")
    return [
        {"kind": "table", "path": str(csv_path), "format": "csv"},
        {"kind": "structured", "path": str(json_path), "format": "json"},
    ]


def _filter_frame(frame: pd.DataFrame, *, dataset_name: str, model_name: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[(frame["dataset_name"] == dataset_name) & (frame["model_name"] == model_name)]


def _compact_dataset_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in (
            "dataset_name",
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "top_generator",
            "top_weight",
            "important_generators",
            "condition_number_max",
        )
        if column in frame.columns
    ]
    return frame[columns]


def _render_metric_summary(metrics: dict[str, dict[str, Any]]) -> str:
    if not metrics:
        return "No aggregate metric table was found."
    frame = pd.DataFrame(
        [
            {"metric": metric, **values}
            for metric, values in metrics.items()
        ]
    )
    return _frame_to_markdown(frame)


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    normalized = frame.copy()
    columns = [str(column) for column in normalized.columns]
    rows = [
        [_format_markdown_cell(value) for value in row]
        for row in normalized.itertuples(index=False, name=None)
    ]
    header = "| " + " | ".join(_escape_markdown_cell(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_sanitize(payload), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _save_figure_bundle(figure, path_without_suffix: Path) -> list[dict[str, str]]:
    artifacts = []
    for extension in ("png", "svg"):
        path = path_without_suffix.with_suffix(f".{extension}")
        figure.savefig(path, dpi=200, bbox_inches="tight")
        artifacts.append({"kind": "plot", "path": str(path), "format": extension})
    return artifacts


def _format_markdown_cell(value: Any, *, max_width: int = 140) -> str:
    if value is None:
        text = ""
    elif isinstance(value, float):
        text = f"{value:.6g}"
    else:
        text = str(value)
    text = _escape_markdown_cell(text.replace("\n", " "))
    if len(text) > max_width:
        return text[: max_width - 3] + "..."
    return text


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if hasattr(value, "item") and value.__class__.__module__.startswith("numpy"):
        return _sanitize(value.item())
    return value


def _ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_at(values: list[Any], index: int) -> float | None:
    if index >= len(values):
        return None
    return _safe_float(values[index])


def _as_shape(value: Any) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(item) for item in value)


def _top_weighted_name(names: list[str], weights: list[float | None]) -> str:
    if not names:
        return ""
    weighted = [
        (index, name, weight if weight is not None else float("-inf"))
        for index, (name, weight) in enumerate(zip(names, weights))
    ]
    if not weighted:
        return names[0]
    return sorted(weighted, key=lambda item: (-item[2], item[0]))[0][1]


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _series_mean(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.mean()) if not clean.empty else None


def _series_median(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.median()) if not clean.empty else None


def _series_min(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.min()) if not clean.empty else None


def _series_max(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.max()) if not clean.empty else None


def _frame_min(frame: pd.DataFrame, column: str) -> float | None:
    return _series_min(frame[column]) if column in frame.columns and not frame.empty else None


def _frame_max(frame: pd.DataFrame, column: str) -> float | None:
    return _series_max(frame[column]) if column in frame.columns and not frame.empty else None
