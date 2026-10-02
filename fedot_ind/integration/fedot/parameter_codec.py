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
    """Copy parameter values recursively into a read-only mapping.

    See ``freeze_value`` for supported container conversions and copy limits.
    """
    return MappingProxyType({key: freeze_value(item) for key, item in value.items()})


def freeze_value(value: Any) -> Any:
    """Snapshot parameters using read-only mappings, tuples, frozensets, and arrays.

    Distinguish lists from tuples and sets from frozensets so ``thaw_value``
    can restore their original container kinds. Object arrays are shallow copies with a read-only flag;
    objects inside them remain shared. Other values use ``deepcopy``, whose
    errors propagate.
    """
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
    """Copy a parameter snapshot into runtime containers.

    Restore frozen lists and sets to their original kinds, mappings to dicts,
    and arrays to writable copies. Object-array elements remain shared;
    other values use ``deepcopy``, whose errors propagate.
    """
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
