"""Prefix classifiers and separately calibrated TEASER/ECEC policies."""

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import OneClassSVM

from .protocol import run_early, training_inputs


def _flatten(x):
    return x.reshape(len(x), -1)


def _class_order(model, classes):
    return [list(model.classes_).index(label) for label in classes]


@dataclass(frozen=True, eq=False)
class PrefixProbabilities:
    schedule: object
    models: tuple
    out_of_fold: np.ndarray

    def predict_prefix(self, prefix, step):
        model = self.models[step]
        return model.predict_proba(prefix.reshape(1, -1))[0, _class_order(model, self.schedule.classes)]

    def predict_one(self, series, policy):
        trace = []

        def decide(prefix, step):
            trace.append(self.predict_prefix(prefix, step))
            choice = policy.choose(np.asarray(trace), prefix)
            return None if choice is None else self.schedule.classes[choice]

        def fallback(prefix, step):
            return self.schedule.classes[int(self.predict_prefix(prefix, step).argmax())]

        return run_early(series, self.schedule, decide, fallback)


def fit_prefix_probabilities(series, labels, schedule, *, estimator=None, folds=3, seed=0):
    x, y = training_inputs(series, labels, schedule)
    if type(folds) is not int or folds < 2 or any(np.count_nonzero(y == c) < folds for c in schedule.classes):
        raise ValueError("Every class must support each stratified training fold.")
    estimator = LogisticRegression(max_iter=400, random_state=seed) if estimator is None else estimator
    split = tuple(StratifiedKFold(folds, shuffle=True, random_state=seed).split(x, y))
    results = np.empty((len(y), len(schedule.cutoffs), len(schedule.classes)))
    models = []
    for step, cutoff in enumerate(schedule.cutoffs):
        features = _flatten(x[:, :, :cutoff])
        for train, holdout in split:
            model = clone(estimator).fit(features[train], y[train])
            results[holdout, step] = model.predict_proba(features[holdout])[:, _class_order(model, schedule.classes)]
        models.append(clone(estimator).fit(features, y))
    results.setflags(write=False)
    return PrefixProbabilities(schedule, tuple(models), results)


def _calibration(scores, target, classes):
    p = np.asarray(scores, dtype=float)
    if (p.ndim != 3 or not all(p.shape) or p.shape != (len(target), p.shape[1], len(classes))
            or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=2), 1)):
        raise ValueError("Expected calibrated (sample, deadline, class) probabilities.")
    return p, np.asarray([classes.index(label) for label in target])


@dataclass(frozen=True)
class TeaserPolicy:
    gates: tuple
    consecutive: int

    def choose(self, trace, _prefix):
        if len(trace) < self.consecutive:
            return None
        label = int(trace[-1].argmax())
        for step in range(len(trace) - self.consecutive, len(trace)):
            values, gate = trace[step], self.gates[step]
            if values.argmax() != label or gate is None or gate.predict(values[None])[0] != 1:
                return None
        return label


def fit_teaser(scores, target, classes, *, consecutive=2, nu=.05):
    p, y = _calibration(scores, target, classes)
    if type(consecutive) is not int or not 1 <= consecutive <= p.shape[1] or not 0 < nu <= 1:
        raise ValueError("Invalid TEASER consistency or SVM parameters.")
    accepted = (p.argmax(axis=2) == y[:, None])
    return TeaserPolicy(tuple(OneClassSVM(nu=nu).fit(p[accepted[:, t], t]) if accepted[:, t].any()
                              else None for t in range(p.shape[1])), consecutive)


@dataclass(frozen=True, eq=False)
class EcecPolicy:
    errors: np.ndarray
    threshold: float

    def choose(self, trace, _prefix):
        labels = trace.argmax(axis=1)
        label = int(labels[-1])
        steps = np.flatnonzero(labels == label)
        evidence = 1. - np.prod(self.errors[steps, label])
        return label if evidence >= self.threshold else None


def fit_ecec(scores, target, classes, *, threshold=.95):
    p, y = _calibration(scores, target, classes)
    if not 0 < threshold <= 1:
        raise ValueError("Confidence threshold must be between zero and one.")
    predicted = p.argmax(axis=2)
    errors = np.empty(p.shape[1:])
    for label in range(len(classes)):
        selected = predicted == label
        errors[:, label] = (1 + np.sum(selected & (y[:, None] != label), axis=0)) / (2 + selected.sum(axis=0))
    return EcecPolicy(errors, threshold)
