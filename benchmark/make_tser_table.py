"""
Generate a markdown table with aeon TSER dataset metadata.

For each dataset loads train/test splits and records:
  n_train, n_test, n_channels, series_length

Output: benchmark/results/tser_aeon/tser_datasets.md

Usage:
    python benchmark/make_tser_table.py
"""

import os
import sys

import pandas as pd
from aeon.datasets import load_regression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.run_tser_aeon import _AEON_DATASETS, OUTPUT_DIR

OUTPUT_MD = os.path.join(OUTPUT_DIR, "tser_datasets.md")


def dataset_meta(name: str) -> dict:
    X_train, _ = load_regression(name, split="train")
    X_test, _ = load_regression(name, split="test")
    return {
        "dataset": name,
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "n_channels": X_train.shape[1],
        "series_length": X_train.shape[2],
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    datasets = list(dict.fromkeys(_AEON_DATASETS))
    rows = []
    total = len(datasets)
    for i, name in enumerate(datasets, 1):
        print(f"[{i}/{total}] {name} ...", end=" ", flush=True)
        try:
            rows.append(dataset_meta(name))
            print("ok")
        except Exception as exc:
            print(f"ERROR: {exc}")
            rows.append({"dataset": name, "n_train": None, "n_test": None,
                         "n_channels": None, "series_length": None})

    df = pd.DataFrame(rows)

    # Build markdown table without requiring tabulate
    cols = list(df.columns)
    col_widths = [max(len(str(c)), df[c].astype(str).str.len().max()) for c in cols]
    def row_line(vals):
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(vals, col_widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    lines = [row_line(cols), sep] + [row_line(df.iloc[i].tolist()) for i in range(len(df))]
    md = "\n".join(lines)

    print("\n" + md)
    with open(OUTPUT_MD, "w") as f:
        f.write(md + "\n")
    print(f"\nSaved → {OUTPUT_MD}")


if __name__ == "__main__":
    main()
