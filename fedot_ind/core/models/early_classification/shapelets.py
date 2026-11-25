"""Early per-channel template matching on observed prefixes."""

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from .protocol import run_early, training_inputs


def _distances(series, shape):
    windows = np.lib.stride_tricks.sliding_window_view(series, len(shape), axis=-1)
    return np.sqrt(np.mean((windows - shape)**2, axis=-1))


@dataclass(frozen=True, eq=False)
class EarlyShapelet:
    channel: int
    label: object
    values: np.ndarray
    threshold: float
    score: float


def _score_shapelet(shape, channel, label, x, y, cutoffs):
    curves = _distances(x[:, channel, :], shape)
    minimum = curves.min(axis=1)
    candidates = np.unique(minimum)
    correct_class = y == label
    best = None
    for threshold in candidates:
        matched = curves <= threshold
        present = matched.any(axis=1)
        correct = present & correct_class
        if not correct.any():
            continue
        found_at = matched[correct].argmax(axis=1) + len(shape)
        deadlines = np.searchsorted(cutoffs, found_at, side="left")
        f1 = 2 * correct.sum() / (present.sum() + correct_class.sum())
        earliness = np.mean(1 - np.asarray(cutoffs)[deadlines] / (cutoffs[-1] + 1))
        score = float(f1 * earliness)
        if best is None or score > best.score:
            best = EarlyShapelet(channel, label, shape.copy(), float(threshold), score)
    return best


def _select_clusters(candidates, similarity, limit):
    representatives = []
    for width in sorted({len(candidate.values) for candidate in candidates}):
        same_width = [candidate for candidate in candidates if len(candidate.values) == width]
        if len(same_width) == 1:
            representatives.extend(same_width)
            continue
        distances = pdist(np.asarray([item.values for item in same_width])) / np.sqrt(width)
        ids = fcluster(linkage(distances, method="complete"), similarity, criterion="distance")
        for ident in np.unique(ids):
            members = [item for item, member_id in zip(same_width, ids) if member_id == ident]
            representatives.append(max(members, key=lambda item: item.score))
    return sorted(representatives, key=lambda item: -item.score)[:limit]


@dataclass(frozen=True, eq=False)
class ShapeletEarlyClassifier:
    schedule: object
    templates: tuple[EarlyShapelet, ...]
    min_channels: int
    centroids: tuple

    def predict_one(self, series):
        def decide(prefix, _step):
            choices = []
            for label in self.schedule.classes:
                by_channel = {}
                for template in self.templates:
                    if (template.label == label and len(template.values) <= prefix.shape[-1]
                            and _distances(prefix[template.channel], template.values).min() <= template.threshold):
                        by_channel[template.channel] = max(by_channel.get(template.channel, 0.), template.score)
                if len(by_channel) >= self.min_channels:
                    choices.append((len(by_channel), sum(by_channel.values()), label))
            return max(choices, key=lambda item: item[:2])[2] if choices else None

        def fallback(prefix, _step):
            return self.schedule.classes[int(np.argmin([np.mean((centroid - prefix)**2)
                                                        for centroid in self.centroids]))]

        return run_early(series, self.schedule, decide, fallback)


def fit_early_shapelets(series, labels, schedule, *, widths, candidates_per_channel=32,
                        templates_per_group=4, min_channels=1, similarity=0., seed=0):
    """Bounded EDSC-style search; multiple channels implement an explicit voting variant."""
    x, y = training_inputs(series, labels, schedule)
    widths = tuple(widths)
    if (not widths or any(type(w) is not int or not 2 <= w <= x.shape[-1] for w in widths)
            or type(candidates_per_channel) is not int or candidates_per_channel < 1
            or type(templates_per_group) is not int or templates_per_group < 1
            or type(min_channels) is not int or not 1 <= min_channels <= schedule.channels
            or not np.isfinite(similarity) or similarity < 0):
        raise ValueError("Invalid shapelet search configuration.")
    rng = np.random.default_rng(seed)
    kept = []
    for channel in range(schedule.channels):
        for label in schedule.classes:
            selected = np.flatnonzero(y == label)
            candidates = []
            for _ in range(candidates_per_channel):
                row, width = int(rng.choice(selected)), int(rng.choice(widths))
                start = int(rng.integers(x.shape[-1] - width + 1))
                shape = x[row, channel, start:start + width]
                candidates.append(_score_shapelet(shape, channel, label, x, y, schedule.cutoffs))
            kept.extend(_select_clusters([item for item in candidates if item is not None],
                                         similarity, templates_per_group))
    centroids = tuple(x[y == label].mean(axis=0) for label in schedule.classes)
    return ShapeletEarlyClassifier(schedule, tuple(kept), min_channels, centroids)
