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
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import timedelta

import numpy as np
import pandas as pd


from aeon.datasets import load_regression

SMALL = [
    'AcousticContaminationMadrid', 'AluminiumConcentration', 'AppliancesEnergy',
    'BarCrawl6min', 'BinanceCoinSentiment', 'BitcoinSentiment', 
    'BoronConcentration', 'CalciumConcentration', 'CardanoSentiment',
    'ChilledWaterPredictor', 'CopperConcentration', 'Covid19Andalusia',
    'Covid3Month', 'DailyOilGasPrices', 'ElectricityPredictor', 
    'EthereumSentiment', 'FloodModeling1', 'FloodModeling2', 'FloodModeling3',
    'GasSensorArrayAcetone', 'GasSensorArrayEthanol', 'HotwaterPredictor',
    'HouseholdPowerConsumption1', 'HouseholdPowerConsumption2', 'IronConcentration',
    'ManganeseConcentration', 'MetroInterstateTrafficVolume',
    'NaturalGasPricesSentiment', 'MetroInterstateTrafficVolume', 'NaturalGasPricesSentiment',
    'OccupancyDetectionLight', 'WindTurbinePower', 'ZincConcentration',
]

MID = [
    'BeijingIntAirportPM25Quality', 'BenzeneConcentration', 'BIDMC32HR', 'BIDMC32RR',
    'BIDMC32SpO2', 'DhakaHourlyAirQuality', 'IEEEPPG',
    'LiveFuelMoistureContent', 'LPGasMonitoringHomeActivity', 'MadridPM10Quality',
    'MagnesiumConcentration', 'MethaneMonitoringHomeActivity', 'NewsHeadlineSentiment', 
    'NewsTitleSentiment', 'ParkingBirmingham', 'PhosphorusConcentration',
    'PotassiumConcentration', 'PrecipitationAndalusia', 'SierraNevadaMountainsSnow',
    'SodiumConcentration', 'SolarRadiationAndalusia', 'SteamPredictor',
    'SulphurConcentration', 'WaveDataTension', 
]

BIG = [
    'BeijingPM10Quality', 'BeijingPM25Quality', 'ElectricMotorTemperature', 
]

HUGE = [
    'AustraliaRainfall', 'DailyTemperatureLatitude', 'NewsHeadlineSentiment',
    'NewsTitleSentiment', 'PPGDalia', 'VentilatorPressure'
]

ALL_DS = SMALL + MID + BIG + HUGE

ds_map = {
    'small': SMALL,
    'mid': MID,
    'big': BIG,
    'huge': HUGE,
    'all': ALL_DS
}

# _AEON_DATASETS = [
#         "AcousticContaminationMadrid", "AluminiumConcentration", "AppliancesEnergy",
#         "AustraliaRainfall", "BarCrawl6min", "BeijingIntAirportPM25Quality",
#         "BeijingPM10Quality", "BeijingPM25Quality", "BenzeneConcentration",
#         "BIDMC32HR", "BIDMC32RR", "BIDMC32SpO2", "BinanceCoinSentiment",
#         "BitcoinSentiment", "BoronConcentration", "CalciumConcentration",
#         "CardanoSentiment", "ChilledWaterPredictor", "CopperConcentration",
#         "Covid19Andalusia", "Covid3Month", "DailyOilGasPrices",
#         "DailyTemperatureLatitude", "DhakaHourlyAirQuality", "ElectricityPredictor",
#         "ElectricMotorTemperature", "EthereumSentiment", "FloodModeling1",
#         "FloodModeling2", "FloodModeling3", "GasSensorArrayAcetone",
#         "GasSensorArrayEthanol", "HotwaterPredictor", "HouseholdPowerConsumption1",
#         "HouseholdPowerConsumption2", "IEEEPPG", "IronConcentration",
#         "LiveFuelMoistureContent", "LPGasMonitoringHomeActivity", "MadridPM10Quality",
#         "MagnesiumConcentration", "ManganeseConcentration", "MethaneMonitoringHomeActivity",
#         "MetroInterstateTrafficVolume", "NaturalGasPricesSentiment", "NewsHeadlineSentiment",
#         "NewsTitleSentiment", "OccupancyDetectionLight", "ParkingBirmingham",
#         "PhosphorusConcentration", "PotassiumConcentration", "PPGDalia",
#         "PrecipitationAndalusia", "SierraNevadaMountainsSnow", "SodiumConcentration",
#         "SolarRadiationAndalusia", "SteamPredictor", "SulphurConcentration",
#         "TetuanEnergyConsumption", "VentilatorPressure", "WaveDataTension",
#         "WindTurbinePower", "ZincConcentration",
#     ]

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
N_PARALLEL = 1       # datasets evaluated in parallel (separate processes)
DASK_WORKERS = 8     # Dask n_workers per dataset
DASK_THREADS = 4     # Dask threads_per_worker per dataset
AUTOML_TIMEOUT = 10  # AutoML timeout per dataset (minutes)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "tser_aeon")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "tser_results.csv")
METRIC_NAMES = ("rmse", "r2", "mae")
LOG_FILE = os.path.join(OUTPUT_DIR, "experiment.log")

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
        "use_cache": False,
    },
}

logger = logging.getLogger("TSER_benchmark")


def _setup_logging():
    """Configure root logger with both StreamHandler and FileHandler.

    logging.basicConfig() is a no-op if the root logger already has handlers
    (FEDOT/dask add them during import). This function explicitly replaces all
    existing handlers so the FileHandler is always installed.
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(fmt)
    root.addHandler(sh)
    root.addHandler(fh)
    # for _name in ("dask", "distributed", "tornado", "asyncio",
    #               "FEDOT logger", "ApiComposer", "AssumptionsHandler", "DataCacher"):
    #     logging.getLogger(_name).setLevel(logging.WARNING)


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
        df = pd.read_csv(RESULTS_CSV)
        if "dataset" in df.columns:
            return df.set_index("dataset")
        return pd.read_csv(RESULTS_CSV, index_col=0)
    return pd.DataFrame(columns=["dataset"] + list(METRIC_NAMES) + ["elapsed_sec", "status"])


def save_row(results: pd.DataFrame, row: dict) -> pd.DataFrame:
    results.loc[row["dataset"]] = row
    import csv
    fieldnames = ["dataset"] + list(METRIC_NAMES) + ["elapsed_sec", "status"]
    write_header = not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return results


def run_dataset(dataset_name: str) -> dict:
    """Run one dataset in a subprocess. Called by ProcessPoolExecutor."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _setup_logging()

    log = logging.getLogger(f"TSER.{dataset_name}")

    train_data, test_data = load_aeon_dataset(dataset_name)
    log.info(f"[{dataset_name}] train={train_data[0].shape} test={test_data[0].shape}")

    cfg = deepcopy(EXPERIMENT_CONFIG)
    dataset_output = os.path.join(OUTPUT_DIR, dataset_name)
    cfg["compute_config"]["output_folder"] = dataset_output
    cfg["compute_config"]["automl_folder"] = {
        "optimisation_history": os.path.join(dataset_output, "opt_hist"),
        "composition_results": os.path.join(dataset_output, "comp_res"),
    }

    t0 = time.time()
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
    elapsed = time.time() - t0

    row = {"dataset": dataset_name, "status": "ok", "elapsed_sec": round(elapsed, 1)}
    for metric in METRIC_NAMES:
        try:
            row[metric] = float(metrics[metric].iloc[0])
        except Exception:
            row[metric] = np.nan

    log.info(
        f"[{dataset_name}] DONE {elapsed:.0f}s | "
        f"rmse={row.get('rmse', float('nan')):.4f} r2={row.get('r2', float('nan')):.4f}"
    )
    return row


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        required=True,
        default='all',
        choices=['small', 'mid', 'big', 'huge', 'all'],
        help="Datasets to run",
    )
    
    args = parser.parse_args()
    logger.info(f"Dataset selection: {args.datasets}")
    _AEON_DATASETS = ds_map[args.datasets]


    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _setup_logging()

    results = load_results()
    completed = set(results.index.tolist())
    all_datasets = list(dict.fromkeys(_AEON_DATASETS))
    datasets = [d for d in all_datasets if d not in completed]
    total = len(all_datasets)

    logger.info(
        f"Total: {total} datasets | done: {len(completed)} | "
        f"remaining: {len(datasets)} | parallel: {N_PARALLEL}"
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
                    "elapsed_sec": np.nan,
                    **{m: np.nan for m in METRIC_NAMES},
                }
            results = save_row(results, row)

            done = len(results)
            avg = results["elapsed_sec"].dropna().mean() if "elapsed_sec" in results.columns else None
            eta = (f" | ETA ~{timedelta(seconds=int(avg * (total - done)))}"
                   if avg and total > done else "")
            logger.info(f"{'='*55}\n  PROGRESS {done}/{total}{eta}\n{'='*55}")

    ok = results[results["status"] == "ok"]
    failed = results[results["status"] != "ok"]
    avg_rmse = f"{ok['rmse'].mean():.4f}" if len(ok) else "n/a"
    avg_r2 = f"{ok['r2'].mean():.4f}" if len(ok) else "n/a"
    logger.info(
        f"\nFINAL SUMMARY"
        f"\n  Completed : {len(ok)}/{total}"
        f"\n  Failed    : {len(failed)}"
        f"\n  Avg RMSE  : {avg_rmse}"
        f"\n  Avg R2    : {avg_r2}"
        f"\n  Results   : {RESULTS_CSV}"
        f"\n  Log       : {LOG_FILE}"
    )


if __name__ == "__main__":
    main()