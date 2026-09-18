from __future__ import annotations

from examples.real_world_examples.industrial_examples.current_api import build_scenario_context


def build_anomaly_detection_context() -> dict:
    return build_scenario_context("anomaly_detection_normal_behavior")


__all__ = ["build_anomaly_detection_context"]
