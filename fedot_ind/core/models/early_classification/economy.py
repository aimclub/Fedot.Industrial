"""Expected misclassification plus delay costs for nonmyopic early decisions."""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from .confidence import _calibration
from .protocol import training_inputs


def _inputs(schedule, scores, target, mistake_costs, delay_costs):
    p, y = _calibration(scores, target, schedule.classes)
    delay = np.asarray(delay_costs, dtype=float)
    cost = np.asarray(mistake_costs, dtype=float)
    if (delay.shape != (p.shape[1],) or cost.shape != (p.shape[-1], p.shape[-1])
            or len(delay) != len(schedule.cutoffs) or not np.isfinite(cost).all()
            or not np.isfinite(delay).all() or np.any(delay < 0) or np.any(cost < 0) or np.any(np.diff(delay) < 0)):
        raise ValueError("Costs must be finite, nonnegative, and delay costs nondecreasing.")
    return p, cost[y[:, None], p.argmax(axis=2)], delay


def _means(costs, assigned, groups):
    result = np.repeat(costs.mean(axis=0, keepdims=True), groups, axis=0)
    for group in range(groups):
        mask = assigned == group
        if mask.any():
            result[group] = costs[mask].mean(axis=0)
    return result


@dataclass(frozen=True, eq=False)
class ClusterEconomy:
    centers: tuple
    future_costs: tuple
    delay: np.ndarray

    def choose(self, trace, prefix):
        step = len(trace) - 1
        distances = np.linalg.norm(self.centers[step] - prefix, axis=(1, 2))
        # Stable soft assignment; fitting and evaluation use the same feature units.
        weights = np.exp(distances.min() - distances)
        risks = weights @ self.future_costs[step] / weights.sum() + self.delay[step:]
        return int(trace[-1].argmax()) if risks.argmin() == 0 else None


def fit_cluster_economy(series, target, schedule, scores, mistake_costs, delay_costs, *, groups, mode="k", seed=0):
    x, y = training_inputs(series, target, schedule)
    _, costs, delay = _inputs(schedule, scores, y, mistake_costs, delay_costs)
    if type(groups) is not int or not 1 <= groups <= len(x) or mode not in {"k", "multi_k"}:
        raise ValueError("Invalid cluster model or number of clusters.")
    centers, tables = [], []
    global_model = None
    for step, cutoff in enumerate(schedule.cutoffs):
        if mode == "k":
            if global_model is None:
                global_model = KMeans(groups, n_init=10, random_state=seed).fit(x.reshape(len(x), -1))
            model = global_model
            center = model.cluster_centers_.reshape(groups, schedule.channels, -1)[:, :, :cutoff]
        else:
            model = KMeans(groups, n_init=10, random_state=seed).fit(x[:, :, :cutoff].reshape(len(x), -1))
            center = model.cluster_centers_.reshape(groups, schedule.channels, cutoff)
        centers.append(center)
        tables.append(_means(costs[:, step:], model.labels_, groups))
    return ClusterEconomy(tuple(centers), tuple(tables), delay)


@dataclass(frozen=True, eq=False)
class GammaEconomy:
    boundaries: tuple
    future_costs: tuple
    delay: np.ndarray
    transitions: tuple

    def choose(self, trace, _prefix):
        step = len(trace) - 1
        group = np.searchsorted(self.boundaries[step], trace[-1, 1], side="right")
        risks = self.future_costs[step][group] + self.delay[step:]
        return int(trace[-1].argmax()) if risks.argmin() == 0 else None


def fit_gamma_economy(schedule, scores, target, mistake_costs, delay_costs, *, groups, mode="gamma"):
    p, costs, delay = _inputs(schedule, scores, target, mistake_costs, delay_costs)
    if len(schedule.classes) != 2 or type(groups) is not int or groups < 1 or mode not in {"gamma", "lite"}:
        raise ValueError("Gamma Economy needs two classes, a valid mode and positive group count.")
    cuts = tuple(np.quantile(p[:, step, 1], np.arange(1, groups) / groups) for step in range(len(delay)))
    state = [np.searchsorted(boundaries, p[:, step, 1], side="right") for step, boundaries in enumerate(cuts)]
    transitions = []
    if mode == "gamma":
        for a, b in zip(state, state[1:]):
            counts = np.ones((groups, groups))
            np.add.at(counts, (a, b), 1)
            transitions.append(counts / counts.sum(axis=1, keepdims=True))
    expected = []
    step_risk = [_means(costs[:, step:step + 1], ids, groups)[:, 0] for step, ids in enumerate(state)]
    for start, ids in enumerate(state):
        if mode == "lite":
            expected.append(_means(costs[:, start:], ids, groups))
            continue
        mapping = np.eye(groups)
        columns = [step_risk[start]]
        for stop in range(start + 1, len(delay)):
            mapping = mapping @ transitions[stop - 1]
            columns.append(mapping @ step_risk[stop])
        expected.append(np.column_stack(columns))
    return GammaEconomy(cuts, tuple(expected), delay, tuple(transitions))
