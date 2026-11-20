"""Shared TRAIN-only comparison plans, measurements, and report provenance."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from platform import python_version
from subprocess import CalledProcessError, run
from time import perf_counter
import tracemalloc

import numpy as np

from industrial_forecasting.grid import positive_count
from industrial_forecasting.metrics import score_forecast

AIRLINE_EXCLUDED_TAIL = 12


def load_airline_train(dtype=float):
    from sktime.datasets import load_airline

    return load_airline().iloc[:-AIRLINE_EXCLUDED_TAIL].to_numpy(dtype=dtype, copy=True)


@dataclass(frozen=True)
class ForecastComparisonPlan:
    origins: tuple[int, ...]
    horizon: int

    def __post_init__(self):
        object.__setattr__(self, "horizon", positive_count(self.horizon, "horizon"))
        origins = tuple(positive_count(origin, "origin") for origin in self.origins)
        if not origins or origins != tuple(sorted(set(origins))):
            raise ValueError("Forecast origins must be unique and increasing.")
        object.__setattr__(self, "origins", origins)

    def windows(self, length):
        if self.origins[-1] + self.horizon > length:
            raise ValueError("A forecast window crosses the supplied TRAIN boundary.")
        return tuple((slice(0, origin), slice(origin, origin + self.horizon)) for origin in self.origins)


def forecast_rows(train, methods, plan):
    series = np.asarray(train)
    for history_window, validation_window in plan.windows(len(series)):
        expected = series[validation_window]
        for name, forecast in methods.items():
            history = series[history_window].copy()
            prediction, cost = measure(forecast, history, plan.horizon)
            yield {"origin": validation_window.start, "horizon": plan.horizon, "model": name,
                   "rmse": score_forecast("rmse", expected, prediction).value, **cost}


def panel_holdout(target, *, seed, stratified=False, groups=None):
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split

    if groups is None:
        return train_test_split(np.arange(len(target)), test_size=.3, random_state=seed,
                                stratify=target if stratified else None)
    groups = np.asarray(groups)
    if not stratified or groups.shape != np.shape(target):
        raise ValueError("Pass one group identifier per classification record.")
    classes = set(target)
    for fit, evaluate in StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed).split(
        target, target, groups
    ):
        if set(target[fit]) == set(target[evaluate]) == classes:
            return fit, evaluate
    raise ValueError("No group-disjoint split contains all classes in both partitions.")


def measure(operation, *args, trace_memory=False, **kwargs):
    """Time a call; optionally isolate its Python allocation trace."""
    if trace_memory:
        if tracemalloc.is_tracing():
            raise RuntimeError("A comparison cannot take ownership of an active allocation trace.")
        tracemalloc.start()
    started = perf_counter()
    try:
        result = operation(*args, **kwargs)
        cost = {"seconds": perf_counter() - started}
        if trace_memory:
            cost["traced_bytes"] = tracemalloc.get_traced_memory()[1]
        return result, cost
    finally:
        if trace_memory:
            tracemalloc.stop()


@dataclass(frozen=True)
class SourceFingerprint:
    name: str
    sha256: str
    encoding: str

    @classmethod
    def file(cls, path):
        path = Path(path)
        digest = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1048576), b""):
                digest.update(block)
        return cls(path.name, digest.hexdigest(), "file_bytes")

    @classmethod
    def arrays(cls, name, *arrays):
        digest = sha256()
        for values in arrays:
            x = np.ascontiguousarray(values)
            if x.dtype.hasobject:
                raise ValueError("Array fingerprints require a stable byte representation.")
            digest.update(repr((x.shape, x.dtype.str)).encode("ascii"))
            digest.update(x.tobytes())
        return cls(name, digest.hexdigest(), "array_shape_dtype_bytes")


def _revision(root):
    def git(*args):
        return run(("git", *args), cwd=root, check=True, capture_output=True, encoding="utf-8").stdout.strip()

    try:
        return {"revision": git("rev-parse", "HEAD"), "tracked_changes": bool(git("status", "--porcelain", "-uno"))}
    except (OSError, CalledProcessError):
        return {"revision": None, "tracked_changes": None}


def comparison_provenance(sources, distributions, *, repository=None):
    versions = {}
    for name in sorted(set(distributions)):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    root = Path(__file__).resolve().parents[3] if repository is None else repository
    return {**_revision(root), "python": python_version(), "packages": versions,
            "sources": {role: asdict(source) for role, source in sources.items()}}


def comparison_report(experiment, *, provenance, protocol, results):
    return {**results, "schema_version": "industrial_tsf_comparison@1", "experiment": experiment,
            "provenance": provenance, "protocol": protocol}
