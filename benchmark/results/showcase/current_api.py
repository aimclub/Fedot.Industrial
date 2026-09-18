from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from benchmark.industrial import (
    ResultAnalysisSpec,
    build_best_per_dataset_frame,
    build_model_diagnostics_frame,
    load_incremental_run_records,
    load_result_sources,
    render_benchmark_result_analysis_pack,
)
from benchmark.industrial.evaluation.markdown import dataframe_to_markdown


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SHOWCASE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = DEFAULT_SHOWCASE_DIR / "showcase_manifest.json"


@dataclass(frozen=True)
class ResultShowcaseSource:
    key: str
    kind: str
    path: str
    source_label: str
    role: str = "reference"
    metric_name: str = ""
    metric_direction: str = ""
    task_type: str = ""
    dataset_column: str = "dataset_name"
    model_column: str = "model_name"
    value_column: str = "metric_value"
    filters: Mapping[str, Any] | None = None
    model_aliases: Mapping[str, str] | None = None
    include_diagnostics: bool = False
    notes: str = ""

    def as_loader_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "path": self.path,
            "source_label": self.source_label,
            "metric_name": self.metric_name,
            "metric_direction": self.metric_direction,
            "task_type": self.task_type,
            "dataset_column": self.dataset_column,
            "model_column": self.model_column,
            "value_column": self.value_column,
        }
        if self.filters:
            payload["filters"] = dict(self.filters)
        if self.model_aliases:
            payload["model_aliases"] = dict(self.model_aliases)
        return payload


@dataclass(frozen=True)
class ResultShowcaseGroup:
    key: str
    title: str
    task_type: str
    metric_name: str
    metric_direction: str
    sources: tuple[ResultShowcaseSource, ...]
    target_model: str | None = None
    target_source_labels: tuple[str, ...] = ()
    reference_source_labels: tuple[str, ...] = ()
    expected_dataset_count: int | None = None
    owner: str = ""
    last_refreshed: str = ""
    refresh_command: str = ""
    comparison_scope: str = "public_reference_comparison"
    notes: str = ""

    @property
    def spec(self) -> ResultAnalysisSpec:
        return ResultAnalysisSpec(
            metric_name=self.metric_name,
            metric_direction=self.metric_direction,
            source_label=self.key,
            task_type=self.task_type,
        )


@dataclass(frozen=True)
class ResultShowcaseManifest:
    version: str
    title: str
    description: str
    groups: tuple[ResultShowcaseGroup, ...]
    archive_candidates: tuple[Mapping[str, Any], ...]
    owner: str = ""
    last_refreshed: str = ""
    refresh_command: str = ""
    post_merge_checklist: tuple[str, ...] = ()


def load_showcase_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> ResultShowcaseManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(raw, "version", "title", "description", "groups")
    group_defaults = {
        "owner": raw.get("owner", ""),
        "last_refreshed": raw.get("last_refreshed", ""),
        "refresh_command": raw.get("refresh_command", ""),
    }
    groups = tuple(_parse_group({**group_defaults, **item}) for item in raw["groups"])
    return ResultShowcaseManifest(
        version=str(raw["version"]),
        title=str(raw["title"]),
        description=str(raw["description"]),
        groups=groups,
        archive_candidates=tuple(raw.get("archive_candidates", ())),
        owner=str(raw.get("owner", "")),
        last_refreshed=str(raw.get("last_refreshed", "")),
        refresh_command=str(raw.get("refresh_command", "")),
        post_merge_checklist=tuple(str(item) for item in raw.get("post_merge_checklist", ())),
    )


def build_group_result_frame(
        group: ResultShowcaseGroup,
        *,
        project_root: str | Path = REPOSITORY_ROOT,
) -> pd.DataFrame:
    sources = [source.as_loader_dict() for source in group.sources]
    return load_result_sources(sources, project_root=project_root, spec=group.spec)


def render_results_showcase(
        output_dir: str | Path = DEFAULT_SHOWCASE_DIR,
        *,
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        project_root: str | Path = REPOSITORY_ROOT,
) -> Path:
    manifest = load_showcase_manifest(manifest_path)
    root = Path(project_root)
    output_path = Path(output_dir)
    tables_dir = output_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    inventory = build_source_inventory(manifest, project_root=root)
    archive = pd.DataFrame(manifest.archive_candidates)
    overview_rows: list[dict[str, Any]] = []
    current_best_frames: list[pd.DataFrame] = []

    for group in manifest.groups:
        normalized = build_group_result_frame(group, project_root=root)
        diagnostics = _build_group_diagnostics(group, project_root=root)
        source_metadata = inventory[inventory["group_key"] == group.key].copy()
        group_dir = output_path / group.key
        render_benchmark_result_analysis_pack(
            normalized,
            group_dir,
            spec=group.spec,
            target_model=group.target_model,
            expected_dataset_count=group.expected_dataset_count,
            diagnostics_frame=diagnostics,
            source_metadata=source_metadata,
            reference_source_labels=group.reference_source_labels,
            best_target_source_labels=group.target_source_labels,
        )
        overview_rows.append(_build_overview_row(group, normalized, diagnostics))
        current_best_frames.append(_build_current_best_frame(group, normalized))

    overview = pd.DataFrame(overview_rows)
    current_best = (
        pd.concat([frame for frame in current_best_frames if not frame.empty], ignore_index=True)
        if current_best_frames
        else pd.DataFrame()
    )
    _write_table(inventory, tables_dir / "source_inventory")
    _write_table(overview, tables_dir / "benchmark_overview")
    _write_table(current_best, tables_dir / "current_best_per_dataset")
    _write_table(archive, tables_dir / "archive_candidates")

    showcase_manifest = output_path / "showcase_manifest.resolved.json"
    showcase_manifest.write_text(
        json.dumps(
            {
                "version": manifest.version,
                "title": manifest.title,
                "description": manifest.description,
                "owner": manifest.owner,
                "last_refreshed": manifest.last_refreshed,
                "refresh_command": manifest.refresh_command,
                "post_merge_checklist": list(manifest.post_merge_checklist),
                "groups": overview.to_dict(orient="records"),
                "source_count": int(len(inventory)),
                "archive_candidate_count": int(len(archive)),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_showcase_readme(output_path / "README.md", manifest, overview, inventory, archive)
    return showcase_manifest


def build_source_inventory(
        manifest: ResultShowcaseManifest,
        *,
        project_root: str | Path = REPOSITORY_ROOT,
) -> pd.DataFrame:
    root = Path(project_root)
    rows = []
    for group in manifest.groups:
        for source in group.sources:
            source_path = root / source.path
            rows.append(
                {
                    "group_key": group.key,
                    "source_key": source.key,
                    "source_label": source.source_label,
                    "role": source.role,
                    "kind": source.kind,
                    "task_type": source.task_type or group.task_type,
                    "metric_name": source.metric_name or group.metric_name,
                    "metric_direction": source.metric_direction or group.metric_direction,
                    "path": source.path,
                    "exists_locally": source_path.exists(),
                    "file_count": _count_files(source_path),
                    "owner": group.owner,
                    "last_refreshed": group.last_refreshed,
                    "refresh_command": group.refresh_command,
                    "comparison_scope": group.comparison_scope,
                    "notes": source.notes,
                }
            )
    return pd.DataFrame(rows)


def _parse_group(raw: Mapping[str, Any]) -> ResultShowcaseGroup:
    _require(raw, "key", "title", "task_type", "metric_name", "metric_direction", "sources")
    sources = tuple(_parse_source(source, raw) for source in raw["sources"])
    return ResultShowcaseGroup(
        key=str(raw["key"]),
        title=str(raw["title"]),
        task_type=str(raw["task_type"]),
        metric_name=str(raw["metric_name"]),
        metric_direction=str(raw["metric_direction"]),
        sources=sources,
        target_model=str(raw["target_model"]) if raw.get("target_model") else None,
        target_source_labels=tuple(str(item) for item in raw.get("target_source_labels", ())),
        reference_source_labels=tuple(str(item) for item in raw.get("reference_source_labels", ())),
        expected_dataset_count=_optional_int(raw.get("expected_dataset_count")),
        owner=str(raw.get("owner", "")),
        last_refreshed=str(raw.get("last_refreshed", "")),
        refresh_command=str(raw.get("refresh_command", "")),
        comparison_scope=str(raw.get("comparison_scope", "public_reference_comparison")),
        notes=str(raw.get("notes", "")),
    )


def _parse_source(raw: Mapping[str, Any], group: Mapping[str, Any]) -> ResultShowcaseSource:
    _require(raw, "key", "kind", "path", "source_label")
    return ResultShowcaseSource(
        key=str(raw["key"]),
        kind=str(raw["kind"]),
        path=str(raw["path"]),
        source_label=str(raw["source_label"]),
        role=str(raw.get("role", "reference")),
        metric_name=str(raw.get("metric_name", group["metric_name"])),
        metric_direction=str(raw.get("metric_direction", group["metric_direction"])),
        task_type=str(raw.get("task_type", group["task_type"])),
        dataset_column=str(raw.get("dataset_column", "dataset_name")),
        model_column=str(raw.get("model_column", "model_name")),
        value_column=str(raw.get("value_column", "metric_value")),
        filters=raw.get("filters") or None,
        model_aliases=raw.get("model_aliases") or None,
        include_diagnostics=bool(raw.get("include_diagnostics", False)),
        notes=str(raw.get("notes", "")),
    )


def _build_group_diagnostics(
        group: ResultShowcaseGroup,
        *,
        project_root: Path,
) -> pd.DataFrame:
    frames = []
    for source in group.sources:
        if not source.include_diagnostics:
            continue
        records = load_incremental_run_records(project_root / source.path)
        if records.empty:
            continue
        records = records.copy()
        records["source_label"] = source.source_label
        frames.append(records)
    if not frames:
        return build_model_diagnostics_frame(pd.DataFrame())
    return build_model_diagnostics_frame(pd.concat(frames, ignore_index=True))


def _build_overview_row(
        group: ResultShowcaseGroup,
        normalized: pd.DataFrame,
        diagnostics: pd.DataFrame,
) -> dict[str, Any]:
    source_count = normalized["source_label"].nunique() if "source_label" in normalized else 0
    return {
        "group_key": group.key,
        "title": group.title,
        "task_type": group.task_type,
        "metric_name": group.metric_name,
        "metric_direction": group.metric_direction,
        "target_model": group.target_model or "",
        "source_count": int(source_count),
        "dataset_count": int(normalized["dataset_name"].nunique()) if "dataset_name" in normalized else 0,
        "model_count": int(normalized["model_name"].nunique()) if "model_name" in normalized else 0,
        "metric_row_count": int(len(normalized)),
        "diagnostic_row_count": int(len(diagnostics)),
        "owner": group.owner,
        "last_refreshed": group.last_refreshed,
        "refresh_command": group.refresh_command,
        "comparison_scope": group.comparison_scope,
        "notes": group.notes,
    }


def _build_current_best_frame(group: ResultShowcaseGroup, normalized: pd.DataFrame) -> pd.DataFrame:
    if normalized.empty:
        return pd.DataFrame()
    if group.target_source_labels:
        current = normalized[normalized["source_label"].astype(str).isin(group.target_source_labels)].copy()
    elif group.target_model:
        current = normalized[normalized["model_name"].astype(str) == group.target_model].copy()
    else:
        return pd.DataFrame()
    if current.empty:
        return pd.DataFrame()
    best = build_best_per_dataset_frame(current, metric_direction=group.metric_direction)
    if best.empty:
        return best
    best.insert(0, "group_key", group.key)
    best.insert(1, "benchmark_title", group.title)
    return best


def _write_table(frame: pd.DataFrame, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = path_without_suffix.with_suffix(".csv")
    md_path = path_without_suffix.with_suffix(".md")
    frame.to_csv(csv_path, index=False)
    md_path.write_text(dataframe_to_markdown(frame, index=False) if not frame.empty else "No rows.", encoding="utf-8")


def _write_showcase_readme(
        path: Path,
        manifest: ResultShowcaseManifest,
        overview: pd.DataFrame,
        inventory: pd.DataFrame,
        archive: pd.DataFrame,
) -> None:
    lines = [
        f"# {manifest.title}",
        "",
        manifest.description,
        "",
        "This directory is the canonical entrypoint for benchmark result comparisons.",
        "Raw historical folders are kept in place and indexed here instead of being duplicated.",
        "",
        "## Ownership And Freshness",
        "",
        f"Owner: {manifest.owner or 'not specified'}",
        f"Last refreshed: {manifest.last_refreshed or 'not specified'}",
        f"Refresh command: `{manifest.refresh_command or 'not specified'}`",
        "",
        "Post-merge checklist:",
        "",
        *(f"- {item}" for item in manifest.post_merge_checklist),
        "",
        "## Rebuild",
        "",
        "```bash",
        "python -m benchmark.results.showcase",
        "```",
        "",
        "## Benchmark Directions",
        "",
        dataframe_to_markdown(overview, index=False) if not overview.empty else "No benchmark groups.",
        "",
        "## Source Inventory",
        "",
        dataframe_to_markdown(inventory, index=False) if not inventory.empty else "No sources.",
        "",
        "## Archive Candidates",
        "",
        dataframe_to_markdown(archive, index=False) if not archive.empty else "No archive candidates.",
        "",
        "Per-benchmark report packs are stored in sibling folders such as `ucr_classification/` and `tser_regression/`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _count_files(path: Path) -> int:
    if path.is_file():
        return 1
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def _require(raw: Mapping[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in raw]
    if missing:
        raise ValueError(f"Missing required showcase manifest fields: {', '.join(missing)}")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_SHOWCASE_DIR",
    "ResultShowcaseGroup",
    "ResultShowcaseManifest",
    "ResultShowcaseSource",
    "build_group_result_frame",
    "build_source_inventory",
    "load_showcase_manifest",
    "render_results_showcase",
]
