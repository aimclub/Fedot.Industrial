"""
TSER benchmark on all 63 aeon datasets using FedotIndustrial.

Datasets are loaded via the `aeon` package (https://www.aeon-toolkit.org).
Results are saved incrementally to OUTPUT_DIR/tser_results.csv after each dataset.
Already-completed datasets are skipped on re-run.

Usage:
    python benchmark/run_tser_aeon.py

Parallelism tuning (adjust to match the server):
    N_PARALLEL × DASK_WORKERS × DASK_THREADS ≈ total CPU cores
"""

import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# aeon imports
# ---------------------------------------------------------------------------
from aeon.datasets import load_regression

try:
    from aeon.datasets._data_loaders import list_available_datasets
    _AEON_DATASETS = list_available_datasets(task='regression')
except Exception:
    # Fallback: full list from the TSER benchmark paper (Tan et al., 2021)
    _AEON_DATASETS = [
        "AppliancesEnergy", "AustraliaRainfall", "BeijingPM10Quality",
        "BeijingPM25Quality", "BenzeneConcentration", "BIDMC32HR",
        "BIDMC32RR", "BIDMC32SpO2", "Covid3Month", "FloodModeling1",
        "FloodModeling2", "FloodModeling3", "HouseholdPowerConsumption1",
        "HouseholdPowerConsumption2", "IEEEPPG", "LiveFuelMoistureContent",
        "NewsHeadlineSentiment", "NewsTitleSentiment", "PPGDalia",
        "PedestrianCounts", "PowerCons", "SelfRegulationSCP1",
        "SelfRegulationSCP2", "SpokenArabicDigits", "StandWalkJump",
        "UWaveGestureLibrary", "EthanolConcentration", "ERing",
        "BasicMotions", "AtrialFibrillation", "FingerMovements",
        "HandMovementDirection", "Handwriting", "Heartbeat",
        "JapaneseVowels", "Libras", "LSST", "MotorImagery",
        "NATOPS", "PEMS-SF", "PhonemeSpectra", "InsectWingbeat",
        "DuckDuckGeese", "EigenWorms", "Epilepsy", "FaceDetection",
        "RacketSports",
    ]

# ---------------------------------------------------------------------------
# FedotIndustrial imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.core.repository.config_repository import (
    DEFAULT_REG_AUTOML_CONFIG,
    DEFAULT_AUTOML_LEARNING_CONFIG,
    DEFAULT_COMPUTE_CONFIG,
)

# ---------------------------------------------------------------------------
# Configuration — tune to the server
# N_PARALLEL × DASK_WORKERS × DASK_THREADS ≈ total CPU cores
# ---------------------------------------------------------------------------
N_PARALLEL = 4       # datasets evaluated in parallel (separate processes)
DASK_WORKERS = 4     # Dask n_workers per dataset
DASK_THREADS = 2     # Dask threads_per_worker per dataset
AUTOML_TIMEOUT = 10  # AutoML timeout per dataset (minutes)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "tser_aeon")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "tser_results.csv")
METRIC_NAMES = ("rmse", "r2", "mae")

_LEARNING_PARAMS = {
    **DEFAULT_AUTOML_LEARNING_CONFIG,
    "timeout": AUTOML_TIMEOUT,
    "n_jobs": DASK_WORKERS * DASK_THREADS,
}

EXPERIMENT_CONFIG = {
    "industrial_config": {"problem": "regression"},
    "automl_config": DEFAULT_REG_AUTOML_CONFIG,
    "learning_config": {
        "learning_strategy": "from_scratch",
        "learning_strategy_params": _LEARNING_PARAMS,
        "optimisation_loss": {"quality_loss": "rmse"},
    },
    "compute_config": {
        **deepcopy(DEFAULT_COMPUTE_CONFIG),
        "distributed": dict(
            processes=False,
            n_workers=DASK_WORKERS,
            threads_per_worker=DASK_THREADS,
            memory_limit="auto",
        ),
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TSER_benchmark")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_aeon_dataset(dataset_name: str):
    """Load train/test split from aeon.

    aeon returns X as 3D numpy (n_samples, n_channels, n_timepoints).
    FedotIndustrial accepts that format directly.
    """
    X_train, y_train = load_regression(dataset_name, split="train")
    X_test, y_test = load_regression(dataset_name, split="test")
    return (X_train, y_train.astype(float)), (X_test, y_test.astype(float))


def load_results() -> pd.DataFrame:
    if os.path.exists(RESULTS_CSV):
        return pd.read_csv(RESULTS_CSV, index_col=0)
    return pd.DataFrame(columns=["dataset"] + list(METRIC_NAMES) + ["status"])


def save_row(results: pd.DataFrame, row: dict) -> pd.DataFrame:
    results.loc[row["dataset"]] = row
    results.to_csv(RESULTS_CSV)
    return results


def run_dataset(dataset_name: str) -> dict:
    """Run one dataset in a subprocess. Called by ProcessPoolExecutor."""
    # Re-configure logging in subprocess (inherited handler may not work)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger(f"TSER.{dataset_name}")
    log.info(f"=== {dataset_name} started ===")

    train_data, test_data = load_aeon_dataset(dataset_name)

    cfg = deepcopy(EXPERIMENT_CONFIG)
    dataset_output = os.path.join(OUTPUT_DIR, dataset_name)
    cfg["compute_config"]["output_folder"] = dataset_output
    cfg["compute_config"]["automl_folder"] = {
        "optimisation_history": os.path.join(dataset_output, "opt_hist"),
        "composition_results": os.path.join(dataset_output, "comp_res"),
    }

    model = FedotIndustrial(**cfg)
    try:
        model.fit(train_data)
        predictions = model.predict(test_data)
        metrics = model.get_metrics(
            labels=predictions,
            probs=None,
            target=test_data[1],
            metric_names=METRIC_NAMES,
        )
        model.save(mode='all')
    finally:
        model.shutdown()

    row = {"dataset": dataset_name, "status": "ok"}
    for metric in METRIC_NAMES:
        try:
            row[metric] = float(metrics[metric].iloc[0])
        except Exception:
            row[metric] = np.nan

    log.info(f"=== {dataset_name} done: {row} ===")
    return row


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = load_results()
    completed = set(results.index.tolist())
    datasets = [d for d in list(dict.fromkeys(_AEON_DATASETS)) if d not in completed]

    logger.info(
        f"Total datasets: {len(list(dict.fromkeys(_AEON_DATASETS)))}, "
        f"already done: {len(completed)}, remaining: {len(datasets)}, "
        f"running {N_PARALLEL} in parallel"
    )

    with ProcessPoolExecutor(max_workers=N_PARALLEL) as executor:
        futures = {executor.submit(run_dataset, name): name for name in datasets}
        for future in as_completed(futures):
            dataset_name = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                logger.exception(f"{dataset_name} failed: {exc}")
                row = {
                    "dataset": dataset_name,
                    "status": f"error: {exc}",
                    **{m: np.nan for m in METRIC_NAMES},
                }
            results = save_row(results, row)

    logger.info(f"Done. Results saved to {RESULTS_CSV}")
    print(results.to_string())


if __name__ == "__main__":
    main()
