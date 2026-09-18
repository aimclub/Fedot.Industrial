"""Current, reproducible Industrial examples."""

from .benchmark_suites import (
    build_forecasting_suite_config,
    build_kernel_learning_ucr_config_preview,
    build_tsc_suite_config,
    build_tser_suite_config,
    run_all_lightweight_examples,
    run_forecasting_example,
    run_tsc_example,
    run_tser_example,
)

__all__ = [
    "build_forecasting_suite_config",
    "build_kernel_learning_ucr_config_preview",
    "build_tsc_suite_config",
    "build_tser_suite_config",
    "run_all_lightweight_examples",
    "run_forecasting_example",
    "run_tsc_example",
    "run_tser_example",
]
