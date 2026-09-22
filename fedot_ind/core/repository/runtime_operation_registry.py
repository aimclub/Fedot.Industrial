"""Lazy thematic operation registries used by Industrial runtime strategies."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from importlib import import_module
from typing import Callable


class LazyOperationMapping(Mapping[str, type]):
    """Resolve one operation implementation without importing sibling domains."""

    def __init__(
            self,
            targets: Mapping[str, str],
            *,
            fallback: Callable[[str], type] | None = None,
            extra_names: tuple[str, ...] = (),
    ) -> None:
        self._targets = dict(targets)
        self._fallback = fallback
        self._names = tuple(dict.fromkeys((*self._targets, *extra_names)))
        self._resolved: dict[str, type] = {}

    def __getitem__(self, name: str) -> type:
        if name in self._resolved:
            return self._resolved[name]
        target = self._targets.get(name)
        if target is not None:
            module_name, attribute_name = target.split(":", maxsplit=1)
            implementation = getattr(
                import_module(module_name), attribute_name)
        elif self._fallback is not None and name in self._names:
            implementation = self._fallback(name)
        else:
            raise KeyError(name)
        self._resolved[name] = implementation
        return implementation

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)


__all__ = ["LazyOperationMapping"]
