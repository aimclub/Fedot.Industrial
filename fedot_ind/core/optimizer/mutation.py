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


def mutation_weights(actions: Iterable[object], strategy: Iterable[float]) -> list[float]:
    """Align the five FEDOT strategy weights with enabled actions, not their order."""
    weights = tuple(strategy)
    if len(weights) != 5:
        raise ValueError("A FEDOT mutation strategy requires five weights")
    slots = {
        "parameter_change_mutation": 0,
        "single_change": 1, "simple": 1,
        "single_drop": 2, "reduce": 2,
        "single_add": 3, "growth": 3, "local_growth": 3, "tree_growth": 3,
        "single_edge": 4,
    }
    # User callbacks without a structural slot keep an explicit nonzero chance.
    result = [weights[slots[mutation_name(action)]] if mutation_name(action) in slots
              else weights[0] for action in actions]
    total = sum(result)
    return [weight / total for weight in result] if total else result
