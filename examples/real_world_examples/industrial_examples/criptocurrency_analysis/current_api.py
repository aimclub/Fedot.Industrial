from __future__ import annotations

from examples.real_world_examples.industrial_examples.current_api import build_scenario_context


def build_cryptocurrency_contexts() -> tuple[dict, ...]:
    return (
        build_scenario_context("bitcoin_regression"),
        build_scenario_context("ethereum_regression"),
        build_scenario_context("bitcoin_forecasting"),
    )


__all__ = ["build_cryptocurrency_contexts"]
