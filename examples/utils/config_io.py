from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_versioned_json(
        path: str | Path,
        *,
        expected_version: str,
        description: str,
) -> dict[str, Any]:
    payload_path = Path(path)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"{description} root must be a mapping: {payload_path}")
    version = str(payload.get("version", ""))
    if version != expected_version:
        raise ValueError(f"Unsupported {description} version: {version}")
    return payload


__all__ = ["load_versioned_json"]
