from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.industrial.evaluation.kernel_learning import render_kernel_stage1_summary_report

DEFAULT_RUN_DIR = (
        PROJECT_ROOT
        / "benchmark"
        / "results"
        / "kernel_learning"
        / "ucr_two_stage_140526"
        / "kernel_learning_ucr_stage1_ba419d49e4"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a summary report for a kernel-learning stage1 UCR run.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="Path to the stage1 run directory.")
    parser.add_argument("--output-dir", default=None, help="Report output directory. Defaults to <run-dir>/analysis.")
    parser.add_argument("--top-n", type=int, default=12, help="Number of rows to show in markdown report tables.")
    args = parser.parse_args()

    analysis = render_kernel_stage1_summary_report(
        args.run_dir,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )
    report_path = Path(analysis.output_dir or Path(args.run_dir) / "analysis") / "summary_report.md"
    print(f"Report: {report_path}")
    print(f"Datasets: {analysis.summary['dataset_count']}")
    print(f"PSD failures: {analysis.summary['psd_failure_count']}")


if __name__ == "__main__":
    main()
