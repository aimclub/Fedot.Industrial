"""Typed panels and forecasts on a shared, explicit time grid."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .grid import TimeGrid, ordered_index


class ModelScope(str, Enum):
    LOCAL = "local"
    GLOBAL = "global"


class CovariateAvailability(str, Enum):
    OBSERVED = "observed"
    KNOWN_FUTURE = "known_future"


def _series_ids(ids) -> tuple[str, ...]:
    result = tuple(ids) if not isinstance(ids, str) else ()
    if (not result or any(not isinstance(name, str) or not name.strip() for name in result)
            or len(set(result)) != len(result)):
        raise ValueError("Series identifiers must be distinct, nonempty strings.")
    return result


def _values(data, field: str, *, rank: int) -> np.ndarray:
    array = np.array(data, dtype=float, copy=True)
    if rank == 2 and array.ndim == 1:
        array = array[None, :]
    if array.ndim != rank or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{field} must be a nonempty, finite rank-{rank} array.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class Covariate:
    """One named signal, aligned to the scenario axes (series, time)."""

    name: str
    availability: CovariateAvailability
    history: np.ndarray
    future: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Covariate name must be a nonempty string.")
        role = CovariateAvailability(self.availability)
        if role is CovariateAvailability.OBSERVED and self.future is not None:
            raise ValueError(f"Observed covariate {self.name!r} cannot provide future values.")
        if role is CovariateAvailability.KNOWN_FUTURE and self.future is None:
            raise ValueError(f"Known-future covariate {self.name!r} requires future values.")
        object.__setattr__(self, "availability", role)
        object.__setattr__(self, "history", _values(self.history, "Covariate history", rank=2))
        if self.future is not None:
            object.__setattr__(self, "future", _values(self.future, "Covariate future", rank=2))

    def to_dict(self) -> dict:
        return {
            "name": self.name, "availability": self.availability.value,
            "history": self.history.tolist(), "future": None if self.future is None else self.future.tolist(),
        }


@dataclass(frozen=True, eq=False)
class ForecastScenario:
    """Target history and covariates. No future target is part of the input."""

    grid: TimeGrid
    history: np.ndarray
    series_ids: tuple[str, ...] = ("series",)
    scope: ModelScope = ModelScope.LOCAL
    covariates: tuple[Covariate, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.grid, TimeGrid):
            raise TypeError("grid must be a TimeGrid.")
        ids = _series_ids(self.series_ids)
        history = _values(self.history, "Target history", rank=2)
        if history.shape != (len(ids), len(self.grid.history)):
            raise ValueError("Target history must match the series identifiers and history grid.")
        covariates = tuple(self.covariates)
        names = set()
        for item in covariates:
            if not isinstance(item, Covariate) or item.name in names:
                raise ValueError("Provide distinct named Covariates.")
            names.add(item.name)
            if item.history.shape != history.shape:
                raise ValueError(f"Covariate {item.name!r} does not match the target history.")
            if item.future is not None and item.future.shape != (len(ids), self.grid.horizon):
                raise ValueError(f"Covariate {item.name!r} does not cover the forecast grid.")
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "series_ids", ids)
        object.__setattr__(self, "scope", ModelScope(self.scope))
        object.__setattr__(self, "covariates", covariates)

    @property
    def horizon(self) -> int:
        return self.grid.horizon

    def to_dict(self) -> dict:
        return {
            "grid": self.grid.to_dict(), "history": self.history.tolist(),
            "series_ids": list(self.series_ids), "scope": self.scope.value,
            "covariates": [item.to_dict() for item in self.covariates],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ForecastScenario":
        return cls(**{
            **payload,
            "grid": TimeGrid.from_dict(payload["grid"]),
            "covariates": tuple(Covariate(**item) for item in payload["covariates"]),
        })


@dataclass(frozen=True, eq=False)
class PointForecast:
    values: np.ndarray
    future_index: np.ndarray
    series_ids: tuple[str, ...] = ("series",)

    def __post_init__(self) -> None:
        ids = _series_ids(self.series_ids)
        future = ordered_index(self.future_index, "forecast")
        values = _values(self.values, "Forecast", rank=2)
        if values.shape != (len(ids), len(future)):
            raise ValueError("Forecast must match the series identifiers and forecast grid.")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "future_index", future)
        object.__setattr__(self, "series_ids", ids)

    @classmethod
    def for_scenario(cls, scenario: ForecastScenario, values) -> "PointForecast":
        return cls(values, scenario.grid.future, scenario.series_ids)


@dataclass(frozen=True, eq=False)
class QuantileForecast:
    """Predictions of shape (series, horizon, quantile), ordered by level."""

    values: np.ndarray
    levels: tuple[float, ...]
    future_index: np.ndarray
    series_ids: tuple[str, ...] = ("series",)

    def __post_init__(self) -> None:
        ids = _series_ids(self.series_ids)
        future = ordered_index(self.future_index, "forecast")
        levels = _values(self.levels, "Quantile levels", rank=1)
        if not 0 < levels[0] <= levels[-1] < 1 or not (np.diff(levels) > 0).all():
            raise ValueError("Quantile levels must be strictly increasing and belong to (0, 1).")
        values = _values(self.values, "Quantile forecast", rank=3)
        if values.shape != (len(ids), len(future), len(levels)):
            raise ValueError("Quantile forecast must match series identifiers, forecast grid and levels.")
        if (np.diff(values, axis=-1) < 0).any():
            raise ValueError("Quantile predictions must not cross.")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "levels", tuple(levels.tolist()))
        object.__setattr__(self, "future_index", future)
        object.__setattr__(self, "series_ids", ids)

    @classmethod
    def for_scenario(cls, scenario: ForecastScenario, values, levels) -> "QuantileForecast":
        return cls(values, levels, scenario.grid.future, scenario.series_ids)
