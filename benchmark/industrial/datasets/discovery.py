from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .local_io import LocalDatasetParseError, resolve_local_split_paths


def discover_local_ucr_datasets(
        data_root: str | Path,
        *,
        allowed_names: Iterable[str] | None = None,
) -> tuple[str, ...]:
    return discover_local_supervised_datasets(data_root, allowed_names=allowed_names)


def discover_local_supervised_datasets(
        data_root: str | Path,
        *,
        allowed_names: Iterable[str] | None = None,
) -> tuple[str, ...]:
    root = Path(data_root)
    if not root.exists():
        return ()

    allowed = set(allowed_names) if allowed_names is not None else None
    discovered = []
    for dataset_dir in root.iterdir():
        if not dataset_dir.is_dir():
            continue
        name = dataset_dir.name
        if allowed is not None and name not in allowed:
            continue
        try:
            resolve_local_split_paths(name, data_root=root)
        except LocalDatasetParseError:
            continue
        discovered.append(name)
    return tuple(sorted(discovered))
