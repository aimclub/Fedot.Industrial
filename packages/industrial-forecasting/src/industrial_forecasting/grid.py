"""Explicit time coordinates; calendar adapters supply their resolved index."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class GridErrorCode(str, Enum):
    EMPTY = "empty"
    INCOMPARABLE = "incomparable"
    ORDER = "order"
    OVERLAP = "overlap"


class GridError(ValueError):
    def __init__(self, code: GridErrorCode, message: str):
        super().__init__(message)
        self.code = code


def positive_count(value: int, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{field} must be a positive integer.")
    return int(value)


def ordered_index(values, field: str) -> np.ndarray:
    index = np.array(values, copy=True)
    if index.ndim != 1 or not index.size:
        raise GridError(GridErrorCode.EMPTY, f"{field} must be a nonempty one-dimensional index.")
    if index.dtype.kind not in "iufM":
        raise GridError(GridErrorCode.INCOMPARABLE, f"{field} must contain numeric or datetime coordinates.")
    if not np.isfinite(index).all() or not (index[1:] > index[:-1]).all():
        raise GridError(GridErrorCode.ORDER, f"{field} must contain finite, strictly increasing coordinates.")
    index.setflags(write=False)
    return index


def require_future_after_history(history: np.ndarray, future: np.ndarray) -> None:
    """Check a boundary between already ordered axes, including adapter-owned calendar values."""
    if not history.size or not future.size:
        raise GridError(GridErrorCode.EMPTY, "History and forecast coordinates must both be non-empty.")
    try:
        follows = bool(future[0] > history[-1])
    except (TypeError, ValueError) as error:
        raise GridError(GridErrorCode.INCOMPARABLE, "History and forecast coordinates are not comparable.") from error
    if not follows:
        raise GridError(GridErrorCode.OVERLAP, "Forecast coordinates must begin strictly after the history.")


def _index_payload(values: np.ndarray) -> list:
    return values.astype(str).tolist() if values.dtype.kind == "M" else values.tolist()


@dataclass(frozen=True, eq=False)
class TimeGrid:
    """Explicit history and forecast axes, which need not have fixed spacing."""

    history: np.ndarray
    future: np.ndarray

    def __post_init__(self) -> None:
        history = ordered_index(self.history, "history")
        future = ordered_index(self.future, "future")
        if (history.dtype.kind == "M") != (future.dtype.kind == "M"):
            raise GridError(GridErrorCode.INCOMPARABLE, "Both time axes must use the same coordinate kind.")
        require_future_after_history(history, future)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "future", future)

    @property
    def horizon(self) -> int:
        return len(self.future)

    @classmethod
    def positions(cls, history_length: int, horizon: int) -> "TimeGrid":
        start = positive_count(history_length, "history_length")
        return cls(np.arange(start), np.arange(start, start + positive_count(horizon, "horizon")))

    def to_dict(self) -> dict:
        return {
            "history": _index_payload(self.history),
            "future": _index_payload(self.future),
            "kind": "datetime" if self.history.dtype.kind == "M" else "numeric",
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TimeGrid":
        if payload.keys() != {"history", "future", "kind"} or payload["kind"] not in ("numeric", "datetime"):
            raise ValueError("Invalid time grid payload.")
        dtype = "datetime64" if payload["kind"] == "datetime" else None
        grid = cls(np.asarray(payload["history"], dtype=dtype), np.asarray(payload["future"], dtype=dtype))
        if (grid.history.dtype.kind == "M") != (payload["kind"] == "datetime"):
            raise ValueError("Time grid coordinate kind does not match its payload.")
        return grid
