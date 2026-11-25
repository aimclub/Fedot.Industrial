"""Preflight workload estimates for the built-in aeon TSER adapters."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil, isqrt
from numbers import Integral, Real

from .tser import AEON_TYPES


PANEL_BUDGET_POLICY = "tser_panel_budget@1"
DENSE_CELL_LIMIT = 30_000_000
DTW_WORK_LIMIT = 250_000_000
PANEL_WORK_LIMIT = 2_000_000_000


@dataclass(frozen=True)
class PanelShape:
    train: int
    evaluate: int
    channels: int
    length: int

    @property
    def rows(self) -> int:
        return self.train + self.evaluate

    @property
    def cells(self) -> int:
        return self.rows * self.channels * self.length


def panel_shape(train_rows: Sequence[Sequence[float]], evaluation_rows: Sequence[Sequence[float]],
                dimensions: object) -> PanelShape | None:
    """Read the shape of flattened records without allocating a dense array."""
    if not train_rows or not evaluation_rows or type(dimensions) is not int or dimensions <= 0:
        return None
    width = len(train_rows[0])
    if not width or width % dimensions or len(evaluation_rows[0]) != width:
        return None
    return PanelShape(len(train_rows), len(evaluation_rows), dimensions, width // dimensions)


def _positive_count(value):
    return int(value) if isinstance(value, Integral) and not isinstance(value, bool) and value > 0 else None


def _feature_count(value):
    if isinstance(value, str) and value == "minimal":
        return 10
    if value is None or (isinstance(value, str) and value in ("efficient", "comprehensive")):
        return 1024  # Conservative column estimate for the pinned tsfresh presets.
    if isinstance(value, dict) and all(settings is None or isinstance(settings, (tuple, list))
                                       for settings in value.values()):
        return sum(1 if settings is None else len(settings) for settings in value.values()) or None
    return None


def _intervals(value, shape):
    # DrCIF uses the original, differences and periodogram representations.
    # Using the original length for all of them bounds the default formulas.
    rules = value if isinstance(value, (list, tuple)) else (value,)
    per_representation = rules if any(isinstance(rule, (list, tuple)) for rule in rules) else (rules,) * 3
    if len(per_representation) != 3:
        return None
    total = 0
    root = isqrt(shape.channels * shape.length)
    for group in per_representation:
        series_rules = group if isinstance(group, (list, tuple)) else (group,)
        count = 0
        for rule in series_rules:
            if isinstance(rule, str) and rule.lower() == "sqrt":
                count += root
            elif isinstance(rule, str) and rule.lower() == "sqrt-div":
                count += root // 3
            elif isinstance(rule, Integral) and not isinstance(rule, bool):
                count += int(rule)
            else:
                return None
        total += max(1, count)
    return total


def _interval_attributes(value):
    # Catch22 and seven summary statistics; use the largest selection over the three representations.
    if isinstance(value, (tuple, list)):
        counts = tuple(_interval_attributes(item) for item in value)
        return max(counts) if counts and None not in counts else None
    if value is None:
        return 29
    if _positive_count(value) is not None:
        return min(int(value), 29)
    if isinstance(value, Real) and not isinstance(value, bool) and 0 < value <= 1:
        return ceil(value * 29)
    return None


def _model_checks(adapter, shape, params):
    if adapter == "dtw_1nn_regressor":
        distance = params.get("distance", "dtw")
        if not isinstance(distance, str) or distance != "dtw":
            return None
        matrix = shape.length ** 2
        return (("DTW matrix cells", matrix, DENSE_CELL_LIMIT),
                ("DTW alignment work", shape.train * shape.evaluate * shape.channels * matrix, DTW_WORK_LIMIT))
    if adapter == "rocket_ridge_regressor":
        kernels = _positive_count(params.get("n_kernels", 10_000))
        if kernels is None:
            return None
        width = 2 * kernels
        return (("ROCKET transformed cells", shape.rows * width, DENSE_CELL_LIMIT),
                ("ROCKET convolution work", shape.rows * width * shape.channels * shape.length, PANEL_WORK_LIMIT),
                ("RidgeCV decomposition work", shape.train * width * min(shape.train, width), PANEL_WORK_LIMIT))

    if params.get("base_estimator") is not None:
        return None  # The workload of a user-supplied estimator is not known.
    if adapter == "freshprince_regressor":
        features = _feature_count(params.get("default_fc_parameters", "comprehensive"))
        trees = _positive_count(params.get("n_estimators", 200))
        if features is None or trees is None:
            return None
        width = features * shape.channels
        return (("FreshPRINCE transformed cells", shape.rows * width, DENSE_CELL_LIMIT),
                ("FreshPRINCE feature and forest work", shape.rows * width * (shape.length + trees),
                 PANEL_WORK_LIMIT))

    if adapter == "drcif_regressor":
        time_limit = params.get("time_limit_in_minutes")
        if time_limit is not None and not isinstance(time_limit, Real):
            return None
        trees = (_positive_count(params.get("contract_max_n_estimators", 500))
                 if time_limit is not None and time_limit > 0 else _positive_count(params.get("n_estimators", 200)))
        intervals = _intervals(params.get("n_intervals", (4, "sqrt-div")), shape)
        attributes = _interval_attributes(params.get("att_subsample_size", 10))
        if None in (trees, intervals, attributes):
            return None
        width = intervals * attributes
        return (("DrCIF interval matrix cells", shape.rows * width, DENSE_CELL_LIMIT),
                ("DrCIF interval feature work", shape.rows * width * trees * shape.length, PANEL_WORK_LIMIT))
    return None


def panel_budget_rejection(adapter: str, shape: PanelShape | None, params: Mapping[str, object]) -> str | None:
    """Return a reason, empty on admission, or None to retain the legacy guard."""
    if adapter not in AEON_TYPES or shape is None:
        return None
    checks = (("panel input cells", shape.cells, DENSE_CELL_LIMIT),)
    model_checks = _model_checks(adapter, shape, params)
    for name, estimate, limit in (*checks, *(model_checks or ())):
        if estimate > limit:
            return f"Runtime guard skipped {adapter}: {name} estimate {estimate:,} exceeds {limit:,}."
    return "" if model_checks is not None else None
