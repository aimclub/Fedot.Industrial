from __future__ import annotations

from examples.real_world_examples.industrial_examples.current_api import build_scenario_context


def build_environment_monitoring_contexts() -> tuple[dict, ...]:
    return (
        build_scenario_context("environment_forecasting"),
        build_scenario_context("environment_regression"),
    )


__all__ = ["build_environment_monitoring_contexts"]
