"""Packaged real data only. TEST values never determine preprocessing."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Holdout:
    train_x: np.ndarray
    train_y: np.ndarray | None
    test_x: np.ndarray
    test_y: np.ndarray
    test_ids: tuple[str, ...]
    source: dict


def load_holdout(config):
    from sktime.datasets import load_airline, load_basic_motions

    if config.scenario == "airline":
        series = load_airline()
        horizon = config.parameters.horizon
        train, test = series.iloc[:-horizon], series.iloc[-horizon:]
        return Holdout(train.to_numpy(), None, train.to_numpy().copy(), test.to_numpy(),
                       tuple(str(index) for index in test.index),
                       {"loader": "sktime.datasets.load_airline", "dataset": "Airline",
                        "split": f"chronological last {horizon} months", "train_end": str(train.index[-1]),
                        "test_start": str(test.index[0])})
    from sklearn.preprocessing import LabelEncoder

    train_x, train_y = load_basic_motions(split="train", return_type="numpy3d")
    test_x, test_y = load_basic_motions(split="test", return_type="numpy3d")
    labels = LabelEncoder().fit(train_y)
    return Holdout(train_x, labels.transform(train_y), test_x, labels.transform(test_y),
                   tuple(str(index) for index in range(len(test_y))),
                   {"loader": "sktime.datasets.load_basic_motions", "dataset": "BasicMotions",
                    "split": "packaged official TRAIN/TEST", "layout": "sample, channel, time",
                    "class_names": labels.classes_.tolist()})


def normalize_train_only(train, test):
    """Per-channel TRAIN statistics for TS; scalar TRAIN statistics for a series."""
    train, test = np.asarray(train, dtype=float), np.asarray(test, dtype=float)
    if train.ndim not in (1, 3) or train.ndim != test.ndim or not train.size or not test.size:
        raise ValueError(
            "Require nonempty series or sample/channel/time arrays")
    if train.ndim == 3 and train.shape[1:] != test.shape[1:]:
        raise ValueError("TRAIN/TEST channel/time shapes must match")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("Inputs must be finite")
    axes = (0, 2) if train.ndim == 3 else 0
    mean = train.mean(axis=axes, keepdims=True)
    scale = train.std(axis=axes, keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    return ((train - mean) / scale).astype("float32"), ((test - mean) / scale).astype("float32"), {
        "fit_split": "TRAIN", "mean": mean.tolist(), "scale": scale.tolist()}
