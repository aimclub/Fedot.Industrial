from __future__ import annotations

import argparse
import logging
from pathlib import Path

from benchmark.experiments.kernel_learning.configs import KernelLearningTwoStageUCRExperimentConfig
from benchmark.industrial.core import write_json

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    defaults = KernelLearningTwoStageUCRExperimentConfig()
    parser = argparse.ArgumentParser(
        description="Run the two-stage UCR kernel-learning experiment.")
    parser.add_argument(
        "--run-stage-1",
        action="store_true",
        help="Run stage 1 before stage 2. By default stage 1 is loaded from saved artifacts.",
    )
    parser.add_argument(
        "--stage1-run-id",
        default=defaults.stage1_run_id,
        help="Existing stage 1 run id to load when --run-stage-1 is not set. Defaults to latest discovery.",
    )
    parser.add_argument(
        "--stage1-run-policy",
        default=defaults.stage1_run_policy,
        choices=("latest",),
        help="How to resolve stage 1 when --stage1-run-id is omitted.",
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
        stage1_run_policy=args.stage1_run_policy,
        datasets=tuple(args.datasets) if args.datasets is not None else (),
        stage1_output_dir=args.stage1_output_dir,
        stage2_output_dir=args.stage2_output_dir,
        timeout_minutes=args.timeout_minutes,
        pop_size=args.pop_size,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    config = config_from_args(args)

    stage1_result = config.load_or_run_stage1()
    stage1_output = Path(
        stage1_result.config.artifact_spec.output_dir) / stage1_result.run_id
    summary = {
        "stage1_run_id": stage1_result.run_id,
        "stage1_output": str(stage1_output),
        "stage2_output": str(config.stage2_output_dir),
        "run_stage1": bool(args.run_stage_1),
        "skip_stage2": bool(args.skip_stage_2),
        "datasets": list(config.resolve_stage1_dataset_names()) if config.run_stage1 else list(config.datasets),
        "timeout_minutes": config.timeout_minutes,
        "pop_size": config.pop_size,
    }
    logger.info("Stage 1 run ID: %s", stage1_result.run_id)
    logger.info("Stage 1 output: %s", stage1_output)
    if args.skip_stage_2:
        write_json(Path(config.stage2_output_dir) /
                   "two_stage_run_summary.json", summary)
        raise SystemExit(0)

    stage2_result = config.run_stage2(stage1_result)
    summary["stage2_datasets"] = len(stage2_result)
    write_json(Path(config.stage2_output_dir) /
               "two_stage_run_summary.json", summary)
    logger.info("Stage 2 output: %s", config.stage2_output_dir)
    logger.info("Stage 2 datasets: %s", len(stage2_result))


if __name__ == "__main__":
    main()
