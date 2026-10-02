"""Shared immutable snapshots that preserve parameter container types on thaw."""

from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


class _FrozenList(tuple):
    """Distinguish a frozen list from a caller-supplied tuple."""


class _FrozenSet(frozenset):
    """Distinguish a frozen set from a caller-supplied frozenset."""


def freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: freeze_value(item) for key, item in value.items()})


def freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(freeze_value(item) for item in value)
    if isinstance(value, _FrozenList):
        return _FrozenList(freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, set):
        return _FrozenSet(freeze_value(item) for item in value)
    if isinstance(value, _FrozenSet):
        return _FrozenSet(freeze_value(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(freeze_value(item) for item in value)
    if isinstance(value, np.ndarray):
        copied = np.array(value, copy=True)
        if not copied.dtype.hasobject:
            return np.frombuffer(copied.tobytes(), dtype=copied.dtype).reshape(copied.shape)
        copied.setflags(write=False)
        return copied
    return deepcopy(value)


def thaw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw_value(item) for key, item in value.items()}
    if isinstance(value, _FrozenList):
        return [thaw_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(thaw_value(item) for item in value)
    if isinstance(value, _FrozenSet):
        return {thaw_value(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(thaw_value(item) for item in value)
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return deepcopy(value)
