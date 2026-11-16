"""Run the installed forecasting package on an offline monthly series."""

import argparse
from importlib import metadata
from importlib.util import find_spec
import json
from pathlib import Path
import sys

import numpy as np

from industrial_forecasting import ForecastScenario, TimeGrid, score_forecast
from industrial_forecasting.baselines import predict_baseline


def require_isolated_install(neural=False):
    forbidden = ["fedot", "golem", "benchmark"]
    if not neural:
        forbidden.extend(("torch", "fedot_ind"))
    for module in forbidden:
        if find_spec(module) is not None:
            raise RuntimeError(f"Expected a clean environment, found {module}.")
    try:
        metadata.version("fedot-ind")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("The standalone example must not install fedot-ind.")
    dist = metadata.distribution("industrial-forecasting")
    origin = Path(find_spec("industrial_forecasting").origin).resolve()
    target = Path(dist.locate_file("industrial_forecasting/__init__.py")).resolve()
    if origin != target or not origin.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("Forecasting must load from the installed wheel.")


def run_example(neural=False):
    period, horizon = 12, 4
    time = np.arange(72 + horizon)
    values = 20 + 0.15 * time + 3 * np.cos(time * (2 * np.pi / period))
    dates = np.datetime64("2020-01", "M") + time
    scenario = ForecastScenario(TimeGrid(dates[:-horizon], dates[-horizon:]), values[:-horizon])
    predictions = {
        "seasonal_naive": predict_baseline(scenario, "seasonal_naive", seasonal_period=period),
        "linear_trend": predict_baseline(scenario, "linear_trend"),
    }
    if neural:
        import torch
        from industrial_forecasting.neural.models import forecast_series

        torch.set_num_threads(1)
        predictions["lagged_ridge_forecaster"] = forecast_series(
            scenario, "lagged_ridge_forecaster", params={"device": "cpu", "window_size": 16},
        )
        predictions["mssa_mlp"] = forecast_series(
            scenario, "mssa", params={
                "device": "cpu", "window_size": 12, "rank": 2, "head_epochs": 4,
                "head_hidden_layers": 1, "head_hidden_dim": 8, "progress_policy": False,
            },
        )
    results = {}
    for name, prediction in predictions.items():
        np.testing.assert_array_equal(prediction.future_index, scenario.grid.future)
        row = prediction.values[0]
        results[name] = {
            "forecast": row.tolist(),
            "mae": score_forecast("mae", values[-horizon:], row).value,
        }
    return {"horizon": horizon, "forecast_index": dates[-horizon:].astype(str).tolist(), "models": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", action="store_true", help="Include the optional tensor models")
    parser.add_argument("--standalone", action="store_true", help="Require an isolated wheel installation")
    args = parser.parse_args()
    if args.standalone:
        require_isolated_install(args.neural)
    print(json.dumps(run_example(args.neural), indent=2))


if __name__ == "__main__":
    main()
