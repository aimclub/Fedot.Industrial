from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PrefixSchedule:
    cutoffs: tuple[int, ...]
    channels: int
    classes: tuple

    def __post_init__(self):
        if (not isinstance(self.cutoffs, tuple) or not self.cutoffs
                or any(type(i) is not int or i < 1 for i in self.cutoffs)
                or self.cutoffs != tuple(sorted(set(self.cutoffs)))
                or type(self.channels) is not int or self.channels < 1
                or len(set(self.classes)) < 2 or len(set(self.classes)) != len(self.classes)):
            raise ValueError("Use increasing positive cutoffs, positive channels and unique classes.")


@dataclass(frozen=True)
class EarlyDecision:
    label: object
    observed: int
    forced: bool


def run_early(series, schedule, decide, fallback):
    """Callbacks see only a (channels, cutoff) slice and its step index."""
    x = np.asarray(series)
    if x.ndim != 2 or x.shape[0] != schedule.channels or x.shape[1] < schedule.cutoffs[-1]:
        raise ValueError("Series does not support this observation schedule.")
    for step, cutoff in enumerate(schedule.cutoffs):
        prefix = np.asarray(x[:, :cutoff], dtype=float).copy()
        if not np.isfinite(prefix).all():
            raise ValueError("Observed prefix contains nonfinite values.")
        label = decide(prefix, step)
        forced = label is None and step == len(schedule.cutoffs) - 1
        if forced:
            label = fallback(prefix, step)
        if label is not None or forced:
            if label not in schedule.classes:
                raise ValueError("Decision does not have a declared class.")
            return EarlyDecision(label, cutoff, forced)
    raise RuntimeError("Validated schedule did not produce a decision.")


def evaluate_decisions(decisions, targets, lengths, *, delay_cost=None):
    decisions, targets, lengths = tuple(decisions), tuple(targets), np.asarray(lengths, dtype=float)
    if not (len(decisions) == len(targets) == len(lengths)) or not decisions:
        raise ValueError("Require one target and series length for each decision.")
    observed = np.asarray([item.observed for item in decisions])
    if lengths.ndim != 1 or not np.isfinite(lengths).all() or np.any(observed < 1) or np.any(observed > lengths):
        raise ValueError("Decision cutoffs must lie within the observed series.")
    return {"accuracy": float(np.mean([d.label == t for d, t in zip(decisions, targets)])),
            "observed_fraction": float(np.mean(observed / lengths)),
            "delay_cost": None if delay_cost is None else float(np.mean([delay_cost(int(t)) for t in observed]))}


def training_inputs(series, labels, schedule):
    x, y = np.asarray(series, dtype=float), np.asarray(labels)
    if (x.ndim != 3 or x.shape[:2] != (len(y), schedule.channels) or y.ndim != 1
            or x.shape[-1] < schedule.cutoffs[-1] or not np.isfinite(x).all()
            or set(y) != set(schedule.classes)):
        raise ValueError("Expected finite (records, channels, time) and all declared training classes.")
    return x[:, :, :schedule.cutoffs[-1]], y
