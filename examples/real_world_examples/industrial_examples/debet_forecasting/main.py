from __future__ import annotations

from examples.real_world_examples.current_api import debet_forecasting_context


def build_debet_forecasting_context() -> dict:
    return {
        "scenario": "debet_forecasting",
        "context": debet_forecasting_context(),
    }


if __name__ == "__main__":
    print(build_debet_forecasting_context())
