from __future__ import annotations

from examples.real_world_examples.current_api import build_forecasting_benchmark_config, config_summary


def build_m4_forecasting_benchmark(output_dir=None) -> dict:
    config = build_forecasting_benchmark_config(output_dir=output_dir)
    return {"scenario": "m4_forecasting_preview", "benchmark": config_summary(config)}


if __name__ == "__main__":
    print(build_m4_forecasting_benchmark())
