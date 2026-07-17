from __future__ import annotations

from examples.real_world_examples.current_api import skab_context


def build_skab_benchmark_context(folder: str = "valve1") -> dict:
    return {"scenario": "skab_anomaly_detection", "context": skab_context(folder)}


if __name__ == "__main__":
    print(build_skab_benchmark_context())
