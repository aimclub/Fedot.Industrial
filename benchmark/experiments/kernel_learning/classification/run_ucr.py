from __future__ import annotations
from benchmark.experiments.kernel_learning.configs import (
    KernelLearningUCRExperimentConfig,
    print_benchmark_run_bundle,
    run_kernel_learning_suite,
)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    bundle = run_kernel_learning_suite(KernelLearningUCRExperimentConfig.from_env())
    print_benchmark_run_bundle(bundle)


if __name__ == "__main__":
    main()
