"""Snapshot and verify legacy FEDOT imports during the INT migration cycle."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence


LEGACY_MODULES = {
    "fedot.core.data.array_utilities": "fedot.core.data.common.array_utils",
    "fedot.core.data.cv_folds": "fedot.core.data.split.cv_folds",
    "fedot.core.data.data": "fedot.core.data.input_data.data",
    "fedot.core.data.data_split": "fedot.core.data.split.data_split",
    "fedot.core.data.multi_modal": "fedot.core.data.multimodal.multi_modal",
}
SCAN_ROOTS = ("fedot_ind", "benchmark", "examples", "tests")
MAP_PATH = Path("fedot_ind/integration/fedot/import_map.json")


@dataclass(frozen=True, order=True)
class ImportRecord:
    path: str
    module: str
    names: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"path": self.path, "module": self.module, "names": list(self.names)}


def scan_imports(root: Path, scan_roots: Iterable[str] = SCAN_ROOTS) -> tuple[ImportRecord, ...]:
    records = []
    for relative_root in scan_roots:
        directory = root / relative_root
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            relative = path.relative_to(root).as_posix()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module in LEGACY_MODULES:
                    records.append(ImportRecord(
                        relative,
                        node.module,
                        tuple(sorted(alias.name for alias in node.names)),
                    ))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in LEGACY_MODULES:
                            records.append(ImportRecord(relative, alias.name, (alias.name,)))
    return tuple(sorted(records))


def build_import_map(records: Iterable[ImportRecord]) -> dict:
    return {
        "schema_version": 1,
        "owner": "fedot_ind.integration.fedot",
        "removal_stage": "IND-FEDOT-05",
        "replacements": [
            {"legacy": legacy, "target": target}
            for legacy, target in sorted(LEGACY_MODULES.items())
        ],
        "imports": [record.to_dict() for record in records],
    }


def verify_import_map(root: Path, payload: object) -> tuple[str, ...]:
    expected = build_import_map(scan_imports(root))
    if payload == expected:
        return ()
    if not isinstance(payload, dict):
        return ("Import map must be a JSON object.",)
    issues = []
    for field in ("schema_version", "owner", "removal_stage", "replacements", "imports"):
        if payload.get(field) != expected[field]:
            issues.append(f"Field {field!r} differs from the current import snapshot.")
    return tuple(issues)


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    path = args.root / MAP_PATH
    if args.write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(build_import_map(scan_imports(args.root)), indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(path)
        return 0
    try:
        payload = _load(path)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Unable to load {path}: {error}")
        return 1
    issues = verify_import_map(args.root, payload)
    for issue in issues:
        print(issue)
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
