"""Load and validate fusion_over_raw experiment defaults."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULTS_PATH = Path(__file__).with_name('defaults.json')
DEFAULTS_VERSION = 'benchmark_industrial_fusion_over_raw@1'

MVP_FUSIONS = frozenset({'concat', 'gated', 'raw_centered_residual', 'film'})
BOTTLENECK_FUSIONS = frozenset(
    {
        'ordinary_bottleneck',
        'raw_residual_bottleneck',
        'context_only_residual_bottleneck',
    }
)
BASELINE_PROFILES = frozenset({'raw', 'raw_larger', 'stats', 'gaf', 'stft'})


class FusionOverRawConfigError(ValueError):
    """Raised when fusion_over_raw defaults are invalid."""


@lru_cache(maxsize=1)
def load_fusion_over_raw_defaults(path: str | Path = DEFAULTS_PATH) -> dict[str, Any]:
    defaults_path = Path(path)
    payload = json.loads(defaults_path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise FusionOverRawConfigError(
            f'Fusion-over-raw defaults root must be a mapping: {defaults_path}'
        )
    version = str(payload.get('version', ''))
    if version != DEFAULTS_VERSION:
        raise FusionOverRawConfigError(
            f'Unsupported fusion-over-raw defaults version: {version}'
        )
    _validate_defaults(payload)
    return payload


def _validate_defaults(payload: dict[str, Any]) -> None:
    required = (
        'experiment_name',
        'primary_metric',
        'metrics',
        'datasets',
        'seeds',
        'single_view_baselines',
        'fusion_experiments',
        'model',
        'training',
        'validation',
        'representations',
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise FusionOverRawConfigError(
            f'Fusion-over-raw defaults missing keys: {sorted(missing)}'
        )
    if str(payload['primary_metric']) != 'f1_macro':
        raise FusionOverRawConfigError(
            f"Expected primary_metric 'f1_macro', got {payload['primary_metric']!r}"
        )
    if 'f1_macro' not in {str(item) for item in payload['metrics']}:
        raise FusionOverRawConfigError('metrics must include f1_macro')
    if not payload['datasets']:
        raise FusionOverRawConfigError('datasets must be non-empty')
    if not payload['seeds']:
        raise FusionOverRawConfigError('seeds must be non-empty')
