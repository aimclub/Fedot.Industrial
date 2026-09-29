"""Typed configuration for the EVO-01 characterization suite."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from importlib.resources import files
from typing import Any, Mapping


EVOLUTION_BASELINE_SCHEMA = "industrial_evolution_baseline@1"


class EvolutionBaselineTask(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    FORECASTING = "ts_forecasting"


@dataclass(frozen=True)
class EvolutionBaselineScenario:
    name: str
    task: EvolutionBaselineTask
    sample_count: int
    series_length: int
    forecast_horizon: int | None
    quality_metric: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EvolutionBaselineScenario":
        scenario = cls(
            name=str(payload["name"]),
            task=EvolutionBaselineTask(str(payload["task"])),
            sample_count=int(payload["sample_count"]),
            series_length=int(payload["series_length"]),
            forecast_horizon=(
                int(payload["forecast_horizon"])
                if payload.get("forecast_horizon") is not None
                else None
            ),
            quality_metric=str(payload["quality_metric"]),
        )
        scenario.validate()
        return scenario

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "Evolution baseline scenario name cannot be empty")
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive")
        if self.series_length < 8:
            raise ValueError("series_length must be at least 8")
        if self.task is EvolutionBaselineTask.FORECASTING:
            if self.forecast_horizon is None or self.forecast_horizon < 1:
                raise ValueError(
                    "forecasting scenario requires a positive forecast_horizon")
            if self.forecast_horizon >= self.series_length:
                raise ValueError(
                    "forecast_horizon must be shorter than series_length")
        elif self.forecast_horizon is not None:
            raise ValueError(
                "forecast_horizon is only valid for forecasting scenarios")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "task": self.task.value,
            "sample_count": self.sample_count,
            "series_length": self.series_length,
            "forecast_horizon": self.forecast_horizon,
            "quality_metric": self.quality_metric,
        }


@dataclass(frozen=True)
class EvolutionBaselineConfig:
    seed: int
    timeout_minutes: float
    population_size: int
    generations: int
    scenarios: tuple[EvolutionBaselineScenario, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EvolutionBaselineConfig":
        if payload.get("schema_version") != EVOLUTION_BASELINE_SCHEMA:
            raise ValueError(
                f"Unsupported evolution baseline schema: {payload.get('schema_version')!r}"
            )
        config = cls(
            seed=int(payload["seed"]),
            timeout_minutes=float(payload["timeout_minutes"]),
            population_size=int(payload["population_size"]),
            generations=int(payload["generations"]),
            scenarios=tuple(
                EvolutionBaselineScenario.from_mapping(item)
                for item in payload.get("scenarios", ())
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.timeout_minutes <= 0:
            raise ValueError("timeout_minutes must be positive")
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.generations < 1:
            raise ValueError("generations must be positive")
        if not self.scenarios:
            raise ValueError(
                "At least one evolution baseline scenario is required")
        names = tuple(scenario.name for scenario in self.scenarios)
        if len(names) != len(set(names)):
            raise ValueError(
                "Evolution baseline scenario names must be unique")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": EVOLUTION_BASELINE_SCHEMA,
            "seed": self.seed,
            "timeout_minutes": self.timeout_minutes,
            "population_size": self.population_size,
            "generations": self.generations,
            "scenarios": [scenario.to_record() for scenario in self.scenarios],
        }


def load_default_evolution_baseline_config() -> EvolutionBaselineConfig:
    defaults_path = files(__package__).joinpath("defaults.json")
    payload = json.loads(defaults_path.read_text(encoding="utf-8"))
    return EvolutionBaselineConfig.from_mapping(payload)
