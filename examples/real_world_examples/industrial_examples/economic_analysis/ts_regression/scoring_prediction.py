from __future__ import annotations

from examples.real_world_examples.current_api import result_summary, run_classification_benchmark


def run_scoring_prediction_example(output_dir=None) -> dict:
    result = run_classification_benchmark(output_dir=output_dir)
    return {"scenario": "scoring_prediction_classification", "benchmark": result_summary(result)}


if __name__ == "__main__":
    print(run_scoring_prediction_example())
