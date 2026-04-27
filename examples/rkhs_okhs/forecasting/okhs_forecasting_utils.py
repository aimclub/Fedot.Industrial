from __future__ import annotations

from typing import Any


def build_forecaster_params(base_params: dict[str, Any] | None = None, **overrides: Any) -> dict[str, Any]:
    params = dict(base_params or {})
    params.update(overrides)
    if "horizon" in params and "forecast_horizon" not in params:
        params["forecast_horizon"] = params.pop("horizon")
    return params


def extract_training_history(forecaster: Any) -> list[float]:
    dmd_model = getattr(forecaster, "dmd_model", None)
    history = getattr(dmd_model, "training_history_", [])
    return list(history)
