from __future__ import annotations

from examples.real_world_examples.industrial_examples.current_api import build_scenario_context


def build_energy_monitoring_context() -> dict:
    return build_scenario_context("energy_regression")


__all__ = ["build_energy_monitoring_context"]
