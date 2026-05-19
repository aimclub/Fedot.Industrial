from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENT_DATE = "140526"
UCR_DATA_ROOT = PROJECT_ROOT / "data"
UCR_DATASETS = ()
STAGE1_OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "results" / "v2_kernel_learning" / f"ucr_two_stage_{EXPERIMENT_DATE}"
STAGE2_OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "results" / "v2_kernel_learning" / f"ucr_two_stage_optim_{EXPERIMENT_DATE}"
STAGE1_RUN_ID = "kernel_learning_ucr_stage1_ba419d49e4"
RUN_STAGE_1 = False

NON_TOPOLOGICAL_GENERATORS = (
    "quantile_extractor_torch",
    "wavelet_extractor",
    "fourier_extractor",
    "recurrence_extractor",
)
STAGE2_METRICS = ("accuracy", "balanced_accuracy", "f1_macro")
STAGE2_TIMEOUT_MINUTES = 5
STAGE2_POP_SIZE = 5


def load_or_run_stage1():
    from fedot_ind.core.kernel_learning.experiments_api import (
        KernelLearningStage1Runner,
        load_stage1_result_from_artifacts,
        resolve_existing_stage1_run_dir,
    )

    if RUN_STAGE_1:
        return KernelLearningStage1Runner(
            data_root=UCR_DATA_ROOT,
            output_dir=STAGE1_OUTPUT_DIR,
            datasets=UCR_DATASETS,
            generator_names=NON_TOPOLOGICAL_GENERATORS,
            metrics=STAGE2_METRICS,
        ).run()

    run_dir = resolve_existing_stage1_run_dir(
        stage1_output_dir=STAGE1_OUTPUT_DIR,
        run_id=STAGE1_RUN_ID,
    )
    return load_stage1_result_from_artifacts(
        run_dir,
        data_root=UCR_DATA_ROOT,
        fallback_generators=NON_TOPOLOGICAL_GENERATORS,
        fallback_metrics=STAGE2_METRICS,
    )


def run_stage2(stage1):
    from fedot_ind.core.kernel_learning.experiments_api import KernelLearningStage2Runner

    return KernelLearningStage2Runner(
        output_dir=STAGE2_OUTPUT_DIR,
        metrics=STAGE2_METRICS,
        timeout_minutes=STAGE2_TIMEOUT_MINUTES,
        pop_size=STAGE2_POP_SIZE,
    ).run(stage1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the two-stage UCR kernel-learning experiment.")
    parser.add_argument(
        "--run-stage-1",
        action="store_true",
        help="Run stage 1 before stage 2. By default stage 1 is loaded from saved artifacts.",
    )
    parser.add_argument(
        "--stage1-run-id",
        default=STAGE1_RUN_ID,
        help="Existing stage 1 run id to load when --run-stage-1 is not set.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="UCR dataset names for stage 1. Pass no names after the flag to use all local datasets.",
    )
    parser.add_argument(
        "--stage1-output-dir",
        type=Path,
        default=STAGE1_OUTPUT_DIR,
        help="Directory containing or receiving stage 1 runs.",
    )
    parser.add_argument(
        "--stage2-output-dir",
        type=Path,
        default=STAGE2_OUTPUT_DIR,
        help="Directory receiving stage 2 optimization artifacts.",
    )
    parser.add_argument(
        "--timeout-minutes",
        type=int,
        default=STAGE2_TIMEOUT_MINUTES,
        help="FEDOT optimization timeout per dataset for stage 2.",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=STAGE2_POP_SIZE,
        help="FEDOT population size for stage 2.",
    )
    parser.add_argument(
        "--skip-stage-2",
        action="store_true",
        help="Only load or run stage 1 and print its location.",
    )
    args = parser.parse_args()

    RUN_STAGE_1 = bool(args.run_stage_1)
    STAGE1_RUN_ID = args.stage1_run_id
    STAGE1_OUTPUT_DIR = args.stage1_output_dir
    STAGE2_OUTPUT_DIR = args.stage2_output_dir
    STAGE2_TIMEOUT_MINUTES = args.timeout_minutes
    STAGE2_POP_SIZE = args.pop_size
    if args.datasets is not None:
        UCR_DATASETS = tuple(args.datasets)

    stage1_result = load_or_run_stage1()
    print(f"Stage 1 run ID: {stage1_result.run_id}")
    print(f"Stage 1 output: {Path(stage1_result.config.artifact_spec.output_dir) / stage1_result.run_id}")
    if args.skip_stage_2:
        raise SystemExit(0)

    stage2_result = run_stage2(stage1_result)
    print(f"Stage 2 output: {STAGE2_OUTPUT_DIR}")
    print(f"Stage 2 datasets: {len(stage2_result)}")
