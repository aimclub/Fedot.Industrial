from __future__ import annotations

from examples.real_world_examples.industrial_examples.current_api import build_scenario_context


def build_health_monitoring_context() -> dict:
    return build_scenario_context("health_monitoring")


__all__ = ["build_health_monitoring_context"]
