"""Benchmark package root.

The canonical implementation lives in ``benchmark.industrial``. This root
package lazily proxies the public Industrial benchmark API for concise imports.
"""

from benchmark.industrial import __all__ as __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from benchmark import industrial

    value = getattr(industrial, name)
    globals()[name] = value
    return value
