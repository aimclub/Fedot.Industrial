from importlib import import_module
from typing import Any


__all__ = ['fedot_api']
__version__ = "0.5.0"


def __getattr__(name: str) -> Any:
    if name != 'fedot_api':
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        return import_module('fedot')
    except ImportError:  # pragma: no cover - optional dependency for local module-level imports
        return None
