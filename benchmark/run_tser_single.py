"""
Run TSER benchmark on a single aeon dataset.

Result is appended to the shared tser_results.csv (same as run_tser_aeon.py).

Usage:
    python benchmark/run_tser_single.py <DatasetName>

Example:
    python benchmark/run_tser_single.py AppliancesEnergy
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.run_tser_aeon import (
    OUTPUT_DIR,
    METRIC_NAMES,
    _setup_logging,
    load_results,
    run_dataset,
    save_row,
)


def main():
    if len(sys.argv) != 2:
        print("Usage: python benchmark/run_tser_single.py <DatasetName>")
        sys.exit(1)

    dataset_name = sys.argv[1]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _setup_logging()

    import logging
    logger = logging.getLogger("TSER_single")
    logger.info(f"Running single dataset: {dataset_name}")

    try:
        row = run_dataset(dataset_name)
    except Exception as exc:
        logger.exception(f"{dataset_name} failed: {exc}")
        import numpy as np
        row = {
            "dataset": dataset_name,
            "status": f"error: {exc}",
            "elapsed_sec": float("nan"),
            **{m: float("nan") for m in METRIC_NAMES},
        }

    results = load_results()
    results = save_row(results, row)
    logger.info(f"Done. Status: {row['status']} | Results saved to tser_results.csv")


if __name__ == "__main__":
    main()
