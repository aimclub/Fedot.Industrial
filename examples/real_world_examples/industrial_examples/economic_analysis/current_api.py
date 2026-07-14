from __future__ import annotations

from examples.real_world_examples.industrial_examples.current_api import build_scenario_context


def build_economic_contexts() -> tuple[dict, ...]:
    return (
        build_scenario_context("economic_regression"),
        build_scenario_context("economic_forecasting"),
    )


__all__ = ["build_economic_contexts"]
