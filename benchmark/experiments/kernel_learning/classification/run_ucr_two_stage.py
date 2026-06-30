from __future__ import annotations
from benchmark.experiments.kernel_learning.configs import KernelLearningTwoStageUCRExperimentConfig

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    defaults = KernelLearningTwoStageUCRExperimentConfig()
    parser = argparse.ArgumentParser(description="Run the two-stage UCR kernel-learning experiment.")
    parser.add_argument(
        "--run-stage-1",
        action="store_true",
        help="Run stage 1 before stage 2. By default stage 1 is loaded from saved artifacts.",
    )
    parser.add_argument(
        "--stage1-run-id",
        default=defaults.stage1_run_id,
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
        default=defaults.stage1_output_dir,
        help="Directory containing or receiving stage 1 runs.",
    )
    parser.add_argument(
        "--stage2-output-dir",
        type=Path,
        default=defaults.stage2_output_dir,
        help="Directory receiving stage 2 optimization artifacts.",
    )
    parser.add_argument(
        "--timeout-minutes",
        type=int,
        default=defaults.timeout_minutes,
        help="FEDOT optimization timeout per dataset for stage 2.",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=defaults.pop_size,
        help="FEDOT population size for stage 2.",
    )
    parser.add_argument(
        "--skip-stage-2",
        action="store_true",
        help="Only load or run stage 1 and print its location.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> KernelLearningTwoStageUCRExperimentConfig:
    return KernelLearningTwoStageUCRExperimentConfig(
        run_stage1=bool(args.run_stage_1),
        stage1_run_id=args.stage1_run_id,
        datasets=tuple(args.datasets) if args.datasets is not None else (),
        stage1_output_dir=args.stage1_output_dir,
        stage2_output_dir=args.stage2_output_dir,
        timeout_minutes=args.timeout_minutes,
        pop_size=args.pop_size,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)

    stage1_result = config.load_or_run_stage1()
    print(f"Stage 1 run ID: {stage1_result.run_id}")
    print(f"Stage 1 output: {Path(stage1_result.config.artifact_spec.output_dir) / stage1_result.run_id}")
    if args.skip_stage_2:
        raise SystemExit(0)

    stage2_result = config.run_stage2(stage1_result)
    print(f"Stage 2 output: {config.stage2_output_dir}")
    print(f"Stage 2 datasets: {len(stage2_result)}")


if __name__ == "__main__":
    main()
