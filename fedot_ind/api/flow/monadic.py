"""Small, explicit helpers around :mod:`pymonad` flows.

These helpers are intentionally lightweight.  They keep Pymonad available for
value transformations and expected branches, while making side effects visible
through named functions such as ``tap`` instead of anonymous mutation lambdas.
"""

from typing import Any, Callable, Optional

from pymonad.either import Either, Left, Right

Step = Callable[[Any], Any]
Predicate = Callable[[Any], bool]
ErrorMapper = Callable[[Any], BaseException]


def _as_either(value: Any) -> Either:
    """Return ``value`` unchanged when it is already an Either."""
    return value if isinstance(value, Either) else Right(value)


def _left_payload(either_value: Either) -> Any:
    """Extract the best available payload from a Pymonad Left value."""
    monoid = getattr(either_value, "monoid", None)
    if isinstance(monoid, (list, tuple)) and monoid:
        return monoid[0]
    return monoid


def pipe(value: Any, *steps: Step) -> Either:
    """Apply ``steps`` to ``value`` inside an ``Either`` flow.

    ``value`` may be a raw object or an existing ``Either``.  Each step can
    return either a raw value or another monadic value supported by Pymonad.
    """
    result = _as_either(value)
    for step in steps:
        result = result.then(step)
    return result


def tap(effect_fn: Callable[[Any], None]) -> Step:
    """Build a step that runs an explicit effect and returns the input value."""

    def _tap(value: Any) -> Any:
        effect_fn(value)
        return value

    _tap.__name__ = getattr(effect_fn, "__name__", "tap")
    return _tap


def branch(predicate: Predicate, left: Step, right: Step) -> Step:
    """Choose a step based on ``predicate``.

    The ``right`` step is used when the predicate is true; otherwise the
    ``left`` step is used.  This mirrors ``Either`` terminology while keeping
    ordinary Python branching easy to read.
    """

    def _branch(value: Any) -> Any:
        return right(value) if predicate(value) else left(value)

    _branch.__name__ = getattr(predicate, "__name__", "branch")
    return _branch


def unwrap_or_raise(either_value: Either, error_mapper: Optional[ErrorMapper] = None) -> Any:
    """Return the Right value or raise a clear exception for a Left value."""
    if either_value.is_right():
        return either_value.value

    error = _left_payload(either_value)
    if error_mapper is not None:
        raise error_mapper(error)
    if isinstance(error, BaseException):
        raise error
    raise ValueError(str(error))


def to_either(value: Any, predicate: Predicate, error: Any) -> Either:
    """Convert ``value`` into Right/Left based on ``predicate``."""
    return Right(value) if predicate(value) else Left(error)


def named_step(name: str, fn: Step) -> Step:
    """Attach a stable debug name to a flow step."""

    def _named(value: Any) -> Any:
        return fn(value)

    _named.__name__ = name
    _named.step_name = name
    return _named
