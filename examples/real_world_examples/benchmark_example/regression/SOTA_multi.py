from __future__ import annotations

from examples.real_world_examples.current_api import result_summary, run_regression_benchmark


def run_sota_multi_regression(output_dir=None) -> dict:
    result = run_regression_benchmark(output_dir=output_dir)
    return {"scenario": "sota_multi_regression", "benchmark": result_summary(result)}


if __name__ == "__main__":
    print(run_sota_multi_regression())
