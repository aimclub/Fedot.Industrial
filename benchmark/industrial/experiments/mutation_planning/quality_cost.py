"""Pure paired reporting of measured runs, not a training or acceptance benchmark.

Pair indexing is O(n), followed by O(p log p) sorting for input-order-independent
output; O(n) memory retains the evidence. No metric is pooled across datasets.
"""

import argparse
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
from typing import Any, Optional


SCHEMA_VERSION = "industrial_mutation_quality_cost@1"
PAIR_FIELDS = ("scenario", "dataset", "seed", "metric", "budget_id")
_COSTS = ("train_seconds", "predict_seconds", "peak_memory_bytes")
_CONSTRAINTS = {
    "max_train_seconds": ("train_seconds", "planned"),
    "max_predict_seconds": ("predict_seconds", "planned"),
    "max_peak_memory_bytes": ("peak_memory_bytes", "planned"),
    "max_train_ratio": ("train_seconds", "ratio"),
    "max_predict_ratio": ("predict_seconds", "ratio"),
    "max_memory_ratio": ("peak_memory_bytes", "ratio"),
}


def _json_safe(value, path="record", active=None):
    """Reject Python-only values, non-string keys, non-finite floats and cycles."""
    active = set() if active is None else active
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) not in (list, dict) or id(value) in active:
        raise ValueError(f"{path}: expected an acyclic JSON-safe value")
    active.add(id(value))
    items = enumerate(value) if type(value) is list else value.items()
    for key, child in items:
        if type(value) is dict and type(key) is not str:
            raise ValueError(f"{path}: JSON object keys must be strings")
        _json_safe(child, f"{path}.{key}", active)
    active.remove(id(value))


def _number(value, field, *, nonnegative=False):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (nonnegative and value < 0):
        qualifier = "nonnegative " if nonnegative else ""
        raise ValueError(f"{field}: expected a finite {qualifier}number (not bool)")


def validate_run_record(record: dict) -> dict:
    """Return an independent JSON-safe record or raise ValueError.

    A failed run has null quality; its elapsed training time is still required.
    Optional memory is never inferred. Scope is explicit, nullable only when no
    memory was measured. Unknown JSON-safe fields are retained as evidence only.
    """
    if type(record) is not dict:
        raise ValueError("record: expected a JSON object")
    _json_safe(record)
    required = set(PAIR_FIELDS) | {
        "task", "mode", "greater_is_better", "quality", "train_seconds",
        "predict_seconds", "memory_scope", "status",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(f"record: missing fields {missing}")
    if "schema_version" in record and record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("record: unsupported schema_version")
    for field in ("task", "scenario", "dataset", "metric", "budget_id"):
        if type(record[field]) is not str or not record[field].strip():
            raise ValueError(f"{field}: expected a nonempty string")
    if type(record["seed"]) is not int:
        raise ValueError("seed: expected an integer (not bool)")
    if record["mode"] not in ("legacy", "planned"):
        raise ValueError("mode: expected legacy or planned")
    if type(record["greater_is_better"]) is not bool:
        raise ValueError("greater_is_better: expected bool")
    if record["status"] not in ("succeeded", "failed"):
        raise ValueError("status: expected succeeded or failed")
    _number(record["train_seconds"], "train_seconds", nonnegative=True)
    if record["status"] == "succeeded":
        _number(record["quality"], "quality")
        _number(record["predict_seconds"], "predict_seconds", nonnegative=True)
    else:
        if record["quality"] is not None:
            raise ValueError("quality: failed runs must have null quality")
        if record["predict_seconds"] is not None:
            _number(record["predict_seconds"], "predict_seconds", nonnegative=True)
    memory = record.get("peak_memory_bytes")
    scope = record["memory_scope"]
    if scope is not None and (type(scope) is not str or not scope.strip()):
        raise ValueError("memory_scope: expected a nonempty string or null")
    if memory is not None:
        if type(memory) is not int or memory < 0:
            raise ValueError("peak_memory_bytes: expected a nonnegative integer or null")
        if scope is None or scope == "not_measured":
            raise ValueError("memory_scope: measured memory requires an explicit scope")
    if "source" in record and record["source"] is not None and type(record["source"]) is not dict:
        raise ValueError("source: expected a JSON object or null")
    return deepcopy(record)


def _finite_result(operation):
    try:
        result = operation()
        return result if type(result) is int or math.isfinite(result) else None
    except (OverflowError, ZeroDivisionError):
        return None


def _difference(legacy, planned):
    if type(legacy) is not type(planned):
        # Preserve the difference before rounding mixed integer/float operands.
        return _finite_result(lambda: float(Fraction(planned) - Fraction(legacy)))
    return _finite_result(lambda: planned - legacy)


def _cost(legacy, planned, *, comparable=True):
    measured = legacy is not None and planned is not None and comparable
    difference = _difference(legacy, planned) if measured else None
    ratio = _finite_result(lambda: planned / legacy) if measured and legacy != 0 else None
    reason = None
    if not comparable:
        reason = "memory_scope_mismatch"
    elif not measured:
        reason = "missing_measurement"
    elif legacy == 0:
        reason = "zero_baseline"
    elif ratio is None:
        reason = "arithmetic_overflow"
    elif ratio == 0 and planned != 0:
        ratio, reason = None, "arithmetic_underflow"
    return {
        "legacy": legacy, "planned": planned, "difference": difference, "ratio": ratio,
        "assessment": "assessed" if measured and difference is not None else "not_assessed",
        "difference_reason": None if difference is not None else reason or "arithmetic_overflow",
        "ratio_reason": reason,
    }


def _criteria(quality_tolerance, cost_constraints):
    if quality_tolerance is None and cost_constraints is None:
        return None
    if quality_tolerance is None or cost_constraints is None:
        raise ValueError("acceptance requires both quality_tolerance and cost_constraints")
    _number(quality_tolerance, "quality_tolerance", nonnegative=True)
    if type(cost_constraints) is not dict or not cost_constraints:
        raise ValueError("cost_constraints: expected a nonempty object")
    for name, value in cost_constraints.items():
        if name not in _CONSTRAINTS:
            raise ValueError(f"cost_constraints: unsupported constraint {name!r}")
        if name == "max_peak_memory_bytes":
            if type(value) is not int or value < 0:
                raise ValueError("max_peak_memory_bytes: expected a nonnegative integer")
        else:
            _number(value, name, nonnegative=True)
    return {"quality_tolerance": quality_tolerance, "cost_constraints": deepcopy(cost_constraints)}


def _assess(pair, criteria):
    if criteria is None:
        return {"status": "not_assessed", "reason": "descriptive_only", "checks": {}}
    if pair["status"] != "succeeded":
        return {"status": "not_assessed", "reason": pair["status"], "checks": {}}
    improvement = pair["quality"]["planned_improvement"]
    quality_check = None
    if improvement is not None:
        quality = pair["quality"]
        sign = 1 if quality["greater_is_better"] else -1
        exact_improvement = sign * (Fraction(quality["planned"]) - Fraction(quality["legacy"]))
        quality_check = exact_improvement >= -Fraction(criteria["quality_tolerance"])
    checks = {"quality_tolerance": quality_check}
    for name, limit in sorted(criteria["cost_constraints"].items()):
        cost_name, field = _CONSTRAINTS[name]
        cost = pair["costs"][cost_name]
        value = cost[field]
        if value is None:
            checks[name] = None
        elif field == "ratio":
            # Do not accept a slight regression hidden by a rounded ratio of 1.
            checks[name] = Fraction(cost["planned"]) <= Fraction(limit) * Fraction(cost["legacy"])
        else:
            checks[name] = value <= limit
    if any(value is None for value in checks.values()):
        status, reason = "not_assessed", "missing_or_undefined_measurement"
    elif all(checks.values()):
        status, reason = "accepted", "explicit_constraints_satisfied"
    else:
        status, reason = "rejected", "explicit_constraints_violated"
    return {"status": status, "reason": reason, "checks": checks}


def _pair(key, modes, criteria):
    legacy, planned = modes.get("legacy"), modes.get("planned")
    issues = []
    if legacy is None or planned is None:
        status = "missing_legacy" if legacy is None else "missing_planned"
    else:
        issues = [f"{field}_mismatch" for field in ("task", "greater_is_better", "memory_scope")
                  if legacy[field] != planned[field]]
        status = "incompatible" if issues else (
            "succeeded" if all(row["status"] == "succeeded" for row in modes.values()) else "failed")

    def values(field):
        return legacy.get(field) if legacy else None, planned.get(field) if planned else None

    old_quality, new_quality = values("quality")
    improvement = None
    quality_reason = None
    if legacy is None or planned is None:
        quality_reason = "missing_counterpart"
    elif "task_mismatch" in issues or "greater_is_better_mismatch" in issues:
        quality_reason = "incompatible"
    elif old_quality is None or new_quality is None:
        quality_reason = "failed_run"
    if old_quality is not None and new_quality is not None and not any(
            issue in issues for issue in ("task_mismatch", "greater_is_better_mismatch")):
        sign = 1 if planned["greater_is_better"] else -1
        difference = _difference(old_quality, new_quality)
        improvement = None if difference is None else sign * difference
        if improvement is None:
            quality_reason = "arithmetic_overflow"
    pair = {
        "key": dict(zip(PAIR_FIELDS, key)), "status": status, "issues": issues,
        "run_status": {mode: modes[mode]["status"] if mode in modes else "missing"
                       for mode in ("legacy", "planned")},
        "quality": {"legacy": old_quality, "planned": new_quality,
                    "greater_is_better": None if "greater_is_better_mismatch" in issues
                    else (planned or legacy)["greater_is_better"],
                    "planned_improvement": improvement,
                    "assessment": "assessed" if improvement is not None else "not_assessed",
                    "reason": quality_reason},
        "memory_scope": {mode: row["memory_scope"] for mode, row in sorted(modes.items())},
        "costs": {field: _cost(*values(field), comparable=(
            field != "peak_memory_bytes" or "memory_scope_mismatch" not in issues)) for field in _COSTS},
    }
    pair["acceptance"] = _assess(pair, criteria)
    return pair


def summarize_quality_cost(records: list, *, quality_tolerance: Optional[float] = None,
                           cost_constraints: Optional[dict] = None) -> dict[str, Any]:
    """Validate and pair run evidence without mutating inputs or inventing limits.

    Difference in quality is oriented so planned-better is positive. Cost ratios
    are planned/legacy, null at a zero baseline or overflow. Acceptance requires
    both explicit policies, successful compatible pairs and defined constrained
    metrics. It says nothing about runs absent from the supplied evidence.
    """
    if type(records) is not list:
        raise ValueError("records: expected a JSON array")
    criteria = _criteria(quality_tolerance, cost_constraints)
    groups = {}
    validated = []
    for index, record in enumerate(records):
        try:
            row = validate_run_record(record)
        except ValueError as error:
            raise ValueError(f"records[{index}]: {error}") from error
        key = tuple(row[field] for field in PAIR_FIELDS)
        modes = groups.setdefault(key, {})
        if row["mode"] in modes:
            raise ValueError(f"duplicate {row['mode']} record for pair {key!r}")
        modes[row["mode"]] = row
        validated.append(row)
    pairs = [_pair(key, groups[key], criteria) for key in sorted(groups)]
    if criteria is None:
        status, reason = "not_assessed", "descriptive_only"
    elif not pairs or any(pair["acceptance"]["status"] == "not_assessed" for pair in pairs):
        status, reason = "not_assessed", "incomplete_or_incompatible_evidence"
    elif all(pair["acceptance"]["status"] == "accepted" for pair in pairs):
        status, reason = "accepted", "all_supplied_pairs_satisfy_explicit_constraints"
    else:
        status, reason = "rejected", "explicit_constraints_violated"
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "supplied_run_records_only_no_training",
        "record_count": len(validated), "pair_count": len(pairs),
        "status_counts": {name: sum(pair["status"] == name for pair in pairs)
                          for name in ("succeeded", "failed", "missing_legacy", "missing_planned", "incompatible")},
        "acceptance": {"status": status, "reason": reason, "criteria": criteria},
        "pairs": pairs,
        "records": sorted(validated, key=lambda row: tuple(row[field] for field in PAIR_FIELDS) + (row["mode"],)),
        "memory_note": "python_tracemalloc measures Python allocations, not process, native or CUDA memory.",
    }


def render_quality_cost_markdown(report: dict) -> str:
    """Render every pair and cost; undefined values remain explicit nulls."""
    def cell(value):
        if type(value) in (dict, list):
            value = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
        else:
            value = "null" if value is None else str(value)
        return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    lines = ["# Paired Quality and Compute Cost", "", f"Schema: `{SCHEMA_VERSION}`", "",
             f"Acceptance: {report['acceptance']['status']} ({report['acceptance']['reason']}).",
             "Only supplied run records are assessed; this report does not train models.",
             "Unique graph counts are not acceptance evidence. Undefined values are null / not_assessed.",
             report["memory_note"], "",
             "Explicit criteria: `" + json.dumps(report["acceptance"]["criteria"], sort_keys=True) + "`", ""]
    for pair in report["pairs"]:
        lines += ["## " + " / ".join(cell(pair["key"][field]) for field in PAIR_FIELDS), "",
                  f"Status: {pair['status']}; acceptance: {pair['acceptance']['status']}.",
                  "Run status: " + cell(pair["run_status"]),
                  "Issues: " + cell(pair["issues"]),
                  "Memory scopes: " + cell(pair["memory_scope"]),
                  "Quality (positive improvement means planned is better): " + cell(pair["quality"]), "",
                  "| Cost | Legacy | Planned | Planned - legacy | Planned / legacy | Assessment | Notes |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for name, cost in pair["costs"].items():
            notes = cost["ratio_reason"] or cost["difference_reason"]
            lines.append("| " + " | ".join(cell(value) for value in (
                name, cost["legacy"], cost["planned"], cost["difference"], cost["ratio"],
                cost["assessment"], notes)) + " |")
        lines += ["", "Acceptance checks: " + cell(pair["acceptance"]), ""]
    return "\n".join(lines) + "\n"


def _load_json(path):
    def unique_object(items):
        result = {}
        for name, value in items:
            if name in result:
                raise ValueError(f"duplicate JSON object field {name!r}")
            result[name] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def main(argv=None) -> int:
    """Read a schema-tagged records envelope; optionally apply explicit criteria."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="JSON object with schema_version and records")
    parser.add_argument("--criteria", type=Path, help="JSON quality_tolerance and cost_constraints")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args(argv)
    try:
        envelope = _load_json(args.records)
        if type(envelope) is not dict or envelope.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"expected schema_version {SCHEMA_VERSION}")
        _json_safe(envelope, "input")
        criteria = _load_json(args.criteria) if args.criteria else {}
        if type(criteria) is not dict or set(criteria) - {"quality_tolerance", "cost_constraints"}:
            raise ValueError("criteria: expected quality_tolerance and cost_constraints only")
        report = summarize_quality_cost(envelope.get("records"), **criteria)
        serialized = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        if args.output_json:
            args.output_json.write_text(serialized, encoding="utf-8")
        if args.output_markdown:
            args.output_markdown.write_text(render_quality_cost_markdown(report), encoding="utf-8")
        if not args.output_json and not args.output_markdown:
            print(serialized, end="")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
