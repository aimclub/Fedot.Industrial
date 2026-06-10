from __future__ import annotations

from examples.real_world_examples.benchmark_example.rkhs_okhs.forecasting.current_api import (
    build_okhs_forecasting_suite_config,
    config_summary,
)


def build_rkhs_okhs_forecasting_example(output_dir=None) -> dict:
    config = build_okhs_forecasting_suite_config(output_dir=output_dir)
    return {
        "scenario": "rkhs_okhs_forecasting",
        "benchmark": config_summary(config),
        "visualization_note": "Use benchmark.industrial.evaluation.render_publication_pack after running the suite.",
    }


if __name__ == "__main__":
    print(build_rkhs_okhs_forecasting_example())
