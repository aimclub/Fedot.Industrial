"""Pure mutation catalogue helpers shared by evolutionary optimisers."""

from __future__ import annotations

from typing import Iterable, TypeVar


MutationT = TypeVar("MutationT")


def mutation_name(mutation: object) -> str:
    """Return a callable or enum-like mutation name without exception probing."""
    callable_name = getattr(mutation, "__name__", None)
    if callable_name is not None:
        return str(callable_name)
    return str(getattr(mutation, "name", ""))


def without_resample_mutations(mutations: Iterable[MutationT]) -> list[MutationT]:
    """Return a new list without mutations unsupported by time-series graphs."""
    return [mutation for mutation in mutations if "resample" not in mutation_name(mutation)]
