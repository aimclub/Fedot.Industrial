from __future__ import annotations

from examples.real_world_examples.industrial_examples.current_api import build_scenario_context


def build_eeg_context() -> dict:
    return build_scenario_context("eeg_classification")


__all__ = ["build_eeg_context"]
