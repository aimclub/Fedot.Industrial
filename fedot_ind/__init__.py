from importlib import import_module
from pkgutil import extend_path
from typing import Any


__all__ = ['fedot_api']
__version__ = "0.5.0"
__path__ = extend_path(__path__, __name__)


def __getattr__(name: str) -> Any:
    if name != 'fedot_api':
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        return import_module('fedot')
    except ImportError:  # pragma: no cover - optional dependency for local module-level imports
        return None
