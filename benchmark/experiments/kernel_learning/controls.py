from __future__ import annotations

import os
from collections.abc import Sequence


def read_csv_env(name: str) -> tuple[str, ...]:
    raw_value = os.environ.get(name, "")
    return tuple(part.strip() for part in raw_value.split(",") if part.strip())


def read_positive_int_env(name: str, default: int | None = None) -> int | None:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def apply_optional_limit(values: Sequence[str], limit: int | None) -> tuple[str, ...]:
    normalized = tuple(values)
    if limit is None:
        return normalized
    return normalized[:limit]
