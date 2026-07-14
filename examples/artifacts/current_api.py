from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG_PATH = PACKAGE_ROOT / "artifact_catalog.json"
DEFAULT_SHOWCASE_DIR = PACKAGE_ROOT / "showcase"
DEFAULT_CLOUD_BUNDLE_DIR = PACKAGE_ROOT / "cloud_bundle"


@dataclass(frozen=True)
class ArtifactInventoryRow:
    key: str
    title: str
    category: str
    task_type: str
    experiment_family: str
    local_path: str
    inventory_mode: str
    storage_policy: str
    exists_locally: bool
    file_count: int
    total_bytes: int
    summary_count: int
    plot_count: int
    table_count: int
    notebook_count: int
    primary_summary: str
    cloud_path: str
    include_policy: str
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "category": self.category,
            "task_type": self.task_type,
            "experiment_family": self.experiment_family,
            "local_path": self.local_path,
            "inventory_mode": self.inventory_mode,
            "storage_policy": self.storage_policy,
            "exists_locally": self.exists_locally,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "summary_count": self.summary_count,
            "plot_count": self.plot_count,
            "table_count": self.table_count,
            "notebook_count": self.notebook_count,
            "primary_summary": self.primary_summary,
            "cloud_path": self.cloud_path,
            "include_policy": self.include_policy,
            "notes": self.notes,
        }


def load_artifact_catalog(catalog_path: str | Path = DEFAULT_CATALOG_PATH) -> dict[str, Any]:
    with Path(catalog_path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def build_artifact_inventory(
        catalog_path: str | Path = DEFAULT_CATALOG_PATH,
        *,
        repository_root: str | Path = REPOSITORY_ROOT,
) -> list[ArtifactInventoryRow]:
    catalog = load_artifact_catalog(catalog_path)
    root = Path(repository_root)
    rows = []
    for group in catalog.get("groups", []):
        local_path = str(group["local_path"])
        absolute_path = root / local_path
        inventory_mode = str(group.get("inventory_mode", "deep"))
        storage_policy = str(group.get("storage_policy", "manifest_only"))
        files = _collect_files(absolute_path, mode=inventory_mode)
        rows.append(
            ArtifactInventoryRow(
                key=str(group["key"]),
                title=str(group["title"]),
                category=str(group["category"]),
                task_type=str(group["task_type"]),
                experiment_family=str(group.get("experiment_family", "")),
                local_path=local_path, inventory_mode=inventory_mode, storage_policy=storage_policy,
                exists_locally=absolute_path.exists(),
                file_count=len(files),
                total_bytes=sum(path.stat().st_size for path in files),
                summary_count=sum(1 for path in files if path.name == "summary.md"),
                plot_count=sum(1 for path in files if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}),
                table_count=sum(
                    1 for path in files if path.suffix.lower() in {".csv", ".md"} and "tables" in path.parts),
                notebook_count=sum(1 for path in files if path.suffix.lower() == ".ipynb"),
                primary_summary=_find_primary_summary(absolute_path, root, mode=inventory_mode),
                cloud_path=str(group.get("cloud_path", "")),
                include_policy=str(group.get("include_policy", "")),
                notes=str(group.get("notes", "")),))
    return rows


def write_cloud_bundle_manifest(
        output_dir: str | Path = DEFAULT_CLOUD_BUNDLE_DIR,
        *,
        catalog_path: str | Path = DEFAULT_CATALOG_PATH,
        repository_root: str | Path = REPOSITORY_ROOT,
        include_local_index: bool = True,
) -> Path:
    catalog = load_artifact_catalog(catalog_path)
    inventory = build_artifact_inventory(catalog_path, repository_root=repository_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "cloud_bundle_manifest.json"
    local_artifact_records = (
        index_local_artifacts(output_path, catalog_path=catalog_path, repository_root=repository_root)
        if include_local_index
        else []
    )
    local_artifacts_by_group = _summarize_local_artifact_records(local_artifact_records)
    payload = {
        "version": "industrial_examples_cloud_bundle@1",
        "title": catalog.get("title", "IndustrialTS examples artifact bundle"),
        "description": catalog.get("description", ""),
        "owner": catalog.get("owner", ""),
        "last_refreshed": catalog.get("last_refreshed", ""),
        "refresh_command": catalog.get("refresh_command", ""),
        "external_archive_url": catalog.get("external_archive_url", ""),
        "artifact_size_policy": catalog.get("artifact_size_policy", {}),
        "external_data_manifest": catalog.get("external_data_manifest"),
        "bundle_policy": {
            "tracked_lightweight_artifacts": "summaries, plots, markdown tables, CSV tables, manifests, and notebooks",
            "large_raw_data": "DVC or manual cloud upload; do not commit raw datasets/checkpoints/zips",
            "credentials": "never commit OAuth tokens, service account files, or local credentials",
            "size_policy": catalog.get("artifact_size_policy", {}),
        },
        "local_files_manifest": "local_artifacts.json" if include_local_index else "",
        "groups": [
            {
                **row.as_dict(),
                "local_file_count": local_artifacts_by_group.get(row.key, {}).get("file_count", 0),
                "local_total_bytes": local_artifacts_by_group.get(row.key, {}).get("total_bytes", 0),
            }
            for row in inventory
        ],
        "benchmark_showcase": catalog.get("benchmark_showcase", []),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_cloud_bundle_readme(output_path / "README.md", payload)
    return manifest_path


def index_local_artifacts(
        output_dir: str | Path = DEFAULT_CLOUD_BUNDLE_DIR,
        *,
        catalog_path: str | Path = DEFAULT_CATALOG_PATH,
        repository_root: str | Path = REPOSITORY_ROOT,
) -> list[dict[str, Any]]:
    """Build a manifest for artifacts that already live under the bundle."""
    catalog = load_artifact_catalog(catalog_path)
    root = Path(repository_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for group in catalog.get("groups", []):
        storage_policy = str(group.get("storage_policy", "manifest_only"))
        if storage_policy == "manifest_only":
            continue
        local_root = root / str(group["local_path"])
        files = _collect_files(local_root, mode=str(group.get("inventory_mode", "deep")))
        for source_path in files:
            target_path = source_path
            records.append(
                {
                    "group_key": str(group["key"]),
                    "category": str(group.get("category", "")),
                    "storage_policy": storage_policy,
                    "source_path": _relative_path(source_path, root),
                    "bundle_path": _relative_path(target_path, output_path),
                    "size_bytes": source_path.stat().st_size,
                }
            )
    manifest_path = output_path / "local_artifacts.json"
    manifest_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return records


def render_artifact_showcase(
        output_dir: str | Path = DEFAULT_SHOWCASE_DIR,
        *,
        catalog_path: str | Path = DEFAULT_CATALOG_PATH,
        repository_root: str | Path = REPOSITORY_ROOT,
) -> Path:
    catalog = load_artifact_catalog(catalog_path)
    inventory = build_artifact_inventory(catalog_path, repository_root=repository_root)
    root = Path(repository_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_inventory_tables(output_path, inventory)
    index_path = output_path / "index.html"
    index_path.write_text(_build_showcase_html(catalog, inventory, root, output_path), encoding="utf-8")
    return index_path


def _collect_files(path: Path, *, mode: str = "deep") -> list[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    if mode == "manifest":
        patterns = ("README.md", "*.json", "*.yaml", "*.yml", "*.md", "*.csv")
        files: list[Path] = []
        for pattern in patterns:
            files.extend(item for item in path.glob(pattern) if item.is_file())
        for child in path.iterdir():
            if child.is_dir():
                for pattern in patterns:
                    files.extend(item for item in child.glob(pattern) if item.is_file())
        for pattern in ("*/summary.md", "*/*/summary.md", "*/*/aggregate/summary.md"):
            files.extend(item for item in path.glob(pattern) if item.is_file())
        return sorted(set(files))
    if mode == "pack":
        return _collect_report_pack_files(path)
    if mode == "shallow":
        files = [item for item in path.glob("*") if item.is_file()]
        files.extend(item for item in path.glob("*/*") if item.is_file())
        files.extend(item for item in path.glob("*/*/*") if item.is_file())
        return sorted(set(files))
    return [item for item in path.rglob("*") if item.is_file()]


def _collect_report_pack_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    files: list[Path] = []
    pack_roots = [path]
    pack_roots.extend(child for child in path.iterdir() if child.is_dir())
    for pack_root in pack_roots:
        files.extend(_existing_files(pack_root / "summary.md", pack_root / "source_metadata.json"))
        plots_dir = pack_root / "plots"
        tables_dir = pack_root / "tables"
        if plots_dir.is_dir():
            files.extend(item for item in plots_dir.iterdir() if item.is_file()
                         and item.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"})
        if tables_dir.is_dir():
            files.extend(item for item in tables_dir.iterdir() if item.is_file()
                         and item.suffix.lower() in {".csv", ".md", ".json"})
        for nested_name in ("forecast_comparison",):
            nested = pack_root / nested_name
            if nested.is_dir():
                files.extend(_existing_files(nested / "summary.md", nested / "source_metadata.json"))
                nested_plots = nested / "plots"
                nested_tables = nested / "tables"
                if nested_plots.is_dir():
                    files.extend(item for item in nested_plots.iterdir() if item.is_file()
                                 and item.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"})
                if nested_tables.is_dir():
                    files.extend(item for item in nested_tables.iterdir() if item.is_file()
                                 and item.suffix.lower() in {".csv", ".md", ".json"})
    return sorted(set(files))


def _existing_files(*paths: Path) -> list[Path]:
    return [path for path in paths if path.is_file()]


def _find_primary_summary(path: Path, root: Path, *, mode: str = "deep") -> str:
    if path.is_file():
        return _relative_path(path, root)
    if not path.exists():
        return ""
    direct = path / "summary.md"
    if direct.is_file():
        return _relative_path(direct, root)
    if mode in {"manifest", "shallow"}:
        candidates = []
        for pattern in ("*/summary.md", "*/*/summary.md", "*/*/aggregate/summary.md"):
            candidates.extend(path.glob(pattern))
        summaries = sorted(item for item in candidates if item.is_file())
        return _relative_path(summaries[0], root) if summaries else ""
    summaries = sorted(path.rglob("summary.md"))
    return _relative_path(summaries[0], root) if summaries else ""


def _write_inventory_tables(output_path: Path, inventory: Iterable[ArtifactInventoryRow]) -> None:
    rows = [row.as_dict() for row in inventory]
    json_path = output_path / "artifact_inventory.json"
    csv_path = output_path / "artifact_inventory.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_cloud_bundle_readme(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# IndustrialTS Artifact Cloud Bundle", "",
        "This folder is the handoff point for publishing examples artifacts outside git.",
        "It contains a machine-readable manifest and keeps raw-data policy explicit.", "",
        "## Ownership And Size Policy", "", f"Owner: {payload.get('owner') or 'not specified'}",
        f"Last refreshed: {payload.get('last_refreshed') or 'not specified'}",
        f"Refresh command: `{payload.get('refresh_command') or 'not specified'}`",
        f"External archive: {payload.get('external_archive_url') or 'not specified'}",
        f"Max committed file size: {payload.get('artifact_size_policy', {}).get('max_single_committed_file_mb', 'not specified')} MB",
        f"Max committed bundle size: {payload.get('artifact_size_policy', {}).get('max_committed_cloud_bundle_mb', 'not specified')} MB",
        "", "## Rules", "",
        "- Upload large raw datasets, checkpoints, archives, and full benchmark runs through DVC or a manual cloud folder.",
        "- Keep credentials in `.dvc/config.local` or environment variables only.",
        "- Lightweight summaries, plots, notebooks, manifests, and CSV/Markdown tables can be mirrored for review.", "",
        "## Groups", "",
        "| Key | Category | Inventory | Storage | Local path | Cloud path | Files | Size bytes | Local files |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",]
    for group in payload["groups"]:
        lines.append(
            f"| `{group['key']}` | `{group['category']}` | `{group['inventory_mode']}` | "
            f"`{group['storage_policy']}` | `{group['local_path']}` | "
            f"`{group['cloud_path']}` | {group['file_count']} | {group['total_bytes']} | "
            f"{group.get('local_file_count', 0)} |"
        )
    lines.append("")
    lines.append("See `cloud_bundle_manifest.json` for the full catalog.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_showcase_html(
        catalog: dict[str, Any],
        inventory: list[ArtifactInventoryRow],
        repository_root: Path,
        output_path: Path,
) -> str:
    benchmark_cards = "\n".join(
        _benchmark_card(item, repository_root, output_path)
        for item in catalog.get("benchmark_showcase", [])
    )
    inventory_rows = "\n".join(_inventory_row(row, repository_root, output_path) for row in inventory)
    total_files = sum(row.file_count for row in inventory)
    total_bytes = sum(row.total_bytes for row in inventory)
    total_plots = sum(row.plot_count for row in inventory)
    total_tables = sum(row.table_count for row in inventory)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(str(catalog.get("title", "IndustrialTS artifact showcase")))}</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1d242d;
      --muted: #657181;
      --line: #d9dee7;
      --accent: #246b5f;
      --accent-2: #7a4f12;
      --warn: #9a3412;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      letter-spacing: 0;
    }}
    header {{
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; font-weight: 650; }}
    h2 {{ margin: 0 0 14px; font-size: 20px; }}
    h3 {{ margin: 0 0 8px; font-size: 16px; }}
    p {{ margin: 0; color: var(--muted); line-height: 1.45; }}
    main {{ padding: 24px 32px 40px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }}
    .metric, .card, .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .metric {{ padding: 14px 16px; }}
    .metric strong {{ display: block; font-size: 24px; margin-bottom: 4px; }}
    .section {{ padding: 18px; margin-bottom: 24px; overflow: hidden; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
    }}
    .card {{ padding: 14px; display: grid; gap: 10px; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      width: fit-content;
      padding: 2px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      font-size: 12px;
      color: var(--muted);
      background: #fbfcfd;
    }}
    .links {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    a {{ color: var(--accent); text-decoration: none; font-weight: 600; }}
    a:hover {{ text-decoration: underline; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 9px 8px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 650; background: #fbfcfd; }}
    code {{ font-family: Consolas, monospace; font-size: 12px; color: var(--accent-2); }}
    .path {{ overflow-wrap: anywhere; }}
    .missing {{ color: var(--warn); font-weight: 650; }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(str(catalog.get("title", "IndustrialTS artifact showcase")))}</h1>
    <p>{escape(str(catalog.get("description", "")))}</p>
  </header>
  <main>
    <section class="metrics">
      <div class="metric"><strong>{len(inventory)}</strong><span>artifact groups</span></div>
      <div class="metric"><strong>{total_files}</strong><span>files indexed</span></div>
      <div class="metric"><strong>{total_plots}</strong><span>plots</span></div>
      <div class="metric"><strong>{total_tables}</strong><span>tables</span></div>
      <div class="metric"><strong>{total_bytes}</strong><span>bytes indexed</span></div>
    </section>
    <section class="section">
      <h2>Main Benchmark Showcase</h2>
      <div class="grid">{benchmark_cards}</div>
    </section>
    <section class="section">
      <h2>Cloud Bundle Inventory</h2>
      <table>
        <thead>
          <tr>
            <th>Group</th><th>Category</th><th>Inventory</th><th>Storage</th><th>Local path</th><th>Cloud path</th><th>Counts</th><th>Policy</th>
          </tr>
        </thead>
        <tbody>{inventory_rows}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def _benchmark_card(item: dict[str, Any], repository_root: Path, output_path: Path) -> str:
    artifact_path = repository_root / str(item.get("artifact_path", ""))
    summary_path = artifact_path / "summary.md"
    plots = sorted((artifact_path / "plots").glob("*.png")) if (artifact_path / "plots").is_dir() else []
    tables = artifact_path / "tables"
    summary_link = _link_to(summary_path, output_path, "summary") if summary_path.is_file(
    ) else "<span class=\"missing\">missing summary</span>"
    plot_link = _link_to(
        plots[0],
        output_path,
        "first plot") if plots else "<span class=\"missing\">missing plot</span>"
    tables_link = _link_to(tables, output_path, "tables") if tables.is_dir(
    ) else "<span class=\"missing\">missing tables</span>"
    return f"""
        <article class="card">
          <span class="badge">{escape(str(item.get("task_type", "")))}</span>
          <h3>{escape(str(item.get("title", item.get("key", ""))))}</h3>
          <p>{escape(str(item.get("notes", "")))}</p>
          <p>Metric: <code>{escape(str(item.get("primary_metric", "")))}</code></p>
          <div class="links">{summary_link}{plot_link}{tables_link}</div>
        </article>
    """


def _inventory_row(row: ArtifactInventoryRow, repository_root: Path, output_path: Path) -> str:
    local = repository_root / row.local_path
    local_link = _link_to(local, output_path, row.local_path) if local.exists(
    ) else f"<span class=\"missing\">{escape(row.local_path)}</span>"
    counts = f"{row.file_count} files<br>{row.plot_count} plots<br>{row.table_count} tables"
    return f"""
        <tr>
          <td><strong>{escape(row.title)}</strong><br><code>{escape(row.key)}</code></td>
          <td>{escape(row.category)}<br><span class="badge">{escape(row.task_type)}</span></td>
          <td><code>{escape(row.inventory_mode)}</code></td>
          <td><code>{escape(row.storage_policy)}</code></td>
          <td class="path">{local_link}</td>
          <td class="path"><code>{escape(row.cloud_path)}</code></td>
          <td>{counts}</td>
          <td>{escape(row.include_policy)}</td>
        </tr>
    """


def _link_to(path: Path, output_path: Path, label: str) -> str:
    try:
        href = Path(os.path.relpath(path.resolve(), output_path.resolve())).as_posix()
    except ValueError:
        href = path.resolve().as_uri()
    return f"<a href=\"{escape(href)}\">{escape(label)}</a>"


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _summarize_local_artifact_records(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for record in records:
        group_key = str(record["group_key"])
        group_summary = summary.setdefault(group_key, {"file_count": 0, "total_bytes": 0})
        group_summary["file_count"] += 1
        group_summary["total_bytes"] += int(record.get("size_bytes", 0))
    return summary
