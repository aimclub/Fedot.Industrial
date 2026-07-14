from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.experiments.kernel_learning.configs import (  # noqa: E402
    KernelLearningM4ExperimentConfig,
    print_benchmark_run_bundle,
    run_kernel_learning_suite,
)


def main() -> None:
    bundle = run_kernel_learning_suite(KernelLearningM4ExperimentConfig.from_env())
    print_benchmark_run_bundle(bundle)


if __name__ == "__main__":
    main()
