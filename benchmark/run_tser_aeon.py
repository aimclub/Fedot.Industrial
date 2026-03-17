"""
TSER benchmark on all 63 aeon datasets using FedotIndustrial.

Datasets are loaded via the `aeon` package (https://www.aeon-toolkit.org).
Results are saved incrementally to OUTPUT_DIR/tser_results.csv after each dataset.
Already-completed datasets are skipped on re-run.

Usage:
    python benchmark/run_tser_aeon.py
"""

import logging
import os
import sys
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
        "PedestrianCounts", "PowerCons", "RacketSports",
        "SelfRegulationSCP1", "SelfRegulationSCP2", "SpokenArabicDigits",
        "StandWalkJump", "UWaveGestureLibrary", "EthanolConcentration",
        "ERing", "BasicMotions", "AtrialFibrillation", "FingerMovements",
        "HandMovementDirection", "Handwriting", "Heartbeat",
        "JapaneseVowels", "Libras", "LSST", "MotorImagery",
        "NATOPS", "PEMS-SF", "PhonemeSpectra", "RacketSports",
        "InsectWingbeat", "DuckDuckGeese", "EigenWorms", "Epilepsy",
        "EthanolConcentration", "FaceDetection", "FingerMovements",
        "HandMovementDirection", "Handwriting", "Heartbeat",
        "JapaneseVowels", "Libras", "LSST", "MotorImagery",
        "NATOPS", "PEMS-SF", "PhonemeSpectra", "SelfRegulationSCP1",
        "SelfRegulationSCP2", "SpokenArabicDigits", "StandWalkJump",
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
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "tser_aeon")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "tser_results.csv")
METRIC_NAMES = ("rmse", "r2", "mae")

EXPERIMENT_CONFIG = {
    "industrial_config": {"problem": "regression"},
    "automl_config": DEFAULT_REG_AUTOML_CONFIG,
    "learning_config": {
        "learning_strategy": "from_scratch",
        "learning_strategy_params": DEFAULT_AUTOML_LEARNING_CONFIG,
        "optimisation_loss": {"quality_loss": "rmse"},
    },
    "compute_config": deepcopy(DEFAULT_COMPUTE_CONFIG),
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
    """Load train/test split from aeon and return (X_train, y_train), (X_test, y_test).

    aeon returns X as 3D numpy (n_samples, n_channels, n_timepoints).
    FedotIndustrial expects (features, target) tuples; features can be 3D numpy.
    """
    X_train, y_train = load_regression(dataset_name, split="train")
    X_test, y_test = load_regression(dataset_name, split="test")
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)
    return (X_train, y_train), (X_test, y_test)


def load_results() -> pd.DataFrame:
    if os.path.exists(RESULTS_CSV):
        return pd.read_csv(RESULTS_CSV, index_col=0)
    return pd.DataFrame(columns=["dataset"] + list(METRIC_NAMES) + ["status"])


def save_row(results: pd.DataFrame, row: dict) -> pd.DataFrame:
    results.loc[row["dataset"]] = row
    results.to_csv(RESULTS_CSV)
    return results


def run_dataset(dataset_name: str) -> dict:
    logger.info(f"=== {dataset_name} ===")

    train_data, test_data = load_aeon_dataset(dataset_name)

    cfg = deepcopy(EXPERIMENT_CONFIG)
    dataset_output = os.path.join(OUTPUT_DIR, dataset_name)
    cfg["compute_config"]["output_folder"] = dataset_output
    cfg["compute_config"]["automl_folder"] = {
        "optimisation_history": os.path.join(dataset_output, "opt_hist"),
        "composition_results": os.path.join(dataset_output, "comp_res"),
    }

    model = FedotIndustrial(**cfg)
    model.fit(train_data)
    predictions = model.predict(test_data)

    metrics = model.get_metrics(
        labels=predictions,
        probs=None,
        target=test_data[1],
        metric_names=METRIC_NAMES,
    )

    model.save_best_model()
    model.shutdown()

    row = {"dataset": dataset_name, "status": "ok"}
    for metric in METRIC_NAMES:
        try:
            row[metric] = float(metrics[metric].iloc[0])
        except Exception:
            row[metric] = np.nan

    logger.info(f"{dataset_name}: {row}")
    return row


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = load_results()
    completed = set(results.index.tolist())
    datasets = list(dict.fromkeys(_AEON_DATASETS))  # deduplicate, preserve order

    logger.info(f"Total datasets: {len(datasets)}, already done: {len(completed)}")

    for dataset_name in datasets:
        if dataset_name in completed:
            logger.info(f"Skipping {dataset_name} (already completed)")
            continue

        try:
            row = run_dataset(dataset_name)
        except Exception as exc:
            logger.exception(f"{dataset_name} failed: {exc}")
            row = {"dataset": dataset_name, "status": f"error: {exc}",
                   **{m: np.nan for m in METRIC_NAMES}}

        results = save_row(results, row)

    logger.info(f"Done. Results saved to {RESULTS_CSV}")
    print(results.to_string())


if __name__ == "__main__":
    main()
