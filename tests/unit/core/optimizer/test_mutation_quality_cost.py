"""Synthetic evidence only; load the pure sidecar without native parent imports."""

from copy import deepcopy
import importlib.util
from itertools import permutations
import json
from pathlib import Path
import subprocess
import sys

from hypothesis import given, strategies as st
import pytest


MODULE_PATH = Path(__file__).resolve().parents[4] / (
    "benchmark/industrial/experiments/mutation_planning/quality_cost.py"
)
SPEC = importlib.util.spec_from_file_location("mutation_quality_cost_sidecar", MODULE_PATH)
qc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qc)


def record(mode="legacy", **changes):
    row = dict(task="regression", scenario="tiny", dataset="synthetic", seed=42,
               mode=mode, metric="rmse", greater_is_better=False, quality=2.0,
               train_seconds=4.0, predict_seconds=2.0, peak_memory_bytes=None,
               memory_scope=None, status="succeeded", budget_id="same-evaluations-10")
    row.update(changes)
    return row


def criteria(**constraints):
    return dict(quality_tolerance=0.1, cost_constraints=constraints or {"max_train_ratio": 1.0})


def pair(rows=None, **policy):
    return qc.summarize_quality_cost(
        rows if rows is not None else [record(), record("planned", quality=1.5)], **policy
    )["pairs"][0]


def test_report_is_descriptive_by_default_and_preserves_evidence():
    rows = [record(source={"revision": "abc", "nested": [1, None]}),
            record("planned", quality=1.5, unique_graphs=100000)]
    before = deepcopy(rows)
    result = qc.summarize_quality_cost(rows)
    assert rows == before
    assert result["schema_version"] == "industrial_mutation_quality_cost@1"
    assert result["scope"] == "supplied_run_records_only_no_training"
    assert result["acceptance"]["status"] == "not_assessed"
    assert result["acceptance"]["criteria"] is None
    assert result["pairs"][0]["acceptance"]["status"] == "not_assessed"
    assert result["pairs"][0]["quality"]["planned_improvement"] == .5
    assert result["pairs"][0]["costs"]["peak_memory_bytes"]["assessment"] == "not_assessed"
    assert result["pairs"][0]["costs"]["peak_memory_bytes"]["ratio"] is None
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    result["records"][0]["source"]["nested"].append(2)
    assert rows == before


@pytest.mark.parametrize("direction,old,new,expected", [
    (True, .6, .8, .2), (False, 2., 1., 1.), (True, 1., 1., 0.), (False, 1., 2., -1.),
])
def test_quality_direction_and_costs(direction, old, new, expected):
    result = pair([record(quality=old, greater_is_better=direction),
                   record("planned", quality=new, greater_is_better=direction,
                          train_seconds=2., predict_seconds=3.)])
    assert result["quality"]["planned_improvement"] == pytest.approx(expected)
    assert result["costs"]["train_seconds"]["difference"] == -2.
    assert result["costs"]["train_seconds"]["ratio"] == .5
    assert result["costs"]["predict_seconds"]["difference"] == 1.
    assert result["costs"]["predict_seconds"]["ratio"] == 1.5


@pytest.mark.parametrize("field,value", [
    ("quality", None), ("quality", True), ("quality", float("nan")),
    ("quality", float("inf")), ("quality", -float("inf")), ("quality", 10 ** 1000),
    ("train_seconds", True), ("train_seconds", -1), ("train_seconds", None),
    ("predict_seconds", False), ("predict_seconds", -1), ("predict_seconds", None),
    ("peak_memory_bytes", True), ("peak_memory_bytes", -1), ("peak_memory_bytes", 1.5),
    ("seed", True), ("seed", 1.0), ("greater_is_better", 1),
    ("status", "planned"), ("status", True), ("mode", "other"), ("mode", []),
    ("task", " "), ("scenario", ""), ("dataset", None), ("budget_id", ""),
    ("metric", 3), ("memory_scope", " "), ("memory_scope", True),
    ("source", []), ("source", {"x": float("nan")}), ("source", {1: "not a JSON key"}),
    ("source", {"x": (1, 2)}), ("schema_version", "unknown@0"),
])
def test_invalid_record_is_rejected_with_index_and_field(field, value):
    with pytest.raises(ValueError, match="records\\[0\\]"):
        qc.summarize_quality_cost([record(**{field: value})])


@pytest.mark.parametrize("field", ["task", "scenario", "dataset", "seed", "mode", "metric",
                                   "greater_is_better", "quality", "train_seconds",
                                   "predict_seconds", "status", "budget_id", "memory_scope"])
def test_required_fields(field):
    row = record()
    del row[field]
    with pytest.raises(ValueError, match=field):
        qc.validate_run_record(row)


def test_json_cycles_and_nonobjects_are_rejected():
    row = record()
    row["source"] = {"self": row}
    for invalid in (row, None, [], True):
        with pytest.raises(ValueError):
            qc.validate_run_record(invalid)
    with pytest.raises(ValueError, match="JSON array"):
        qc.summarize_quality_cost(iter([record()]))


def test_failed_run_preserves_elapsed_cost_without_fake_quality_or_acceptance():
    failed = record("planned", status="failed", quality=None, predict_seconds=None,
                    source={"error_type": "FitError"})
    result = pair([record(), failed], **criteria())
    assert result["status"] == "failed"
    assert result["run_status"] == {"legacy": "succeeded", "planned": "failed"}
    assert result["quality"]["planned_improvement"] is None
    assert result["quality"]["assessment"] == "not_assessed"
    assert result["costs"]["train_seconds"]["planned"] == 4.
    assert result["costs"]["predict_seconds"]["ratio"] is None
    assert result["acceptance"]["status"] == "not_assessed"
    with pytest.raises(ValueError, match="failed runs"):
        qc.validate_run_record(record(status="failed"))
    assert qc.validate_run_record(record(status="failed", quality=None, predict_seconds=0.))["predict_seconds"] == 0.


@pytest.mark.parametrize("mode,missing", [("legacy", "missing_planned"), ("planned", "missing_legacy")])
def test_missing_counterpart_is_not_acceptance(mode, missing):
    result = qc.summarize_quality_cost([record(mode)], **criteria())
    assert result["pairs"][0]["status"] == missing
    assert result["pairs"][0]["quality"]["assessment"] == "not_assessed"
    assert result["acceptance"]["status"] == "not_assessed"


@pytest.mark.parametrize("field,value", [("scenario", "other"), ("dataset", "other"),
                                         ("seed", 43), ("metric", "mae"), ("budget_id", "larger")])
def test_all_pairing_fields_are_fairness_boundaries(field, value):
    result = qc.summarize_quality_cost([record(), record("planned", **{field: value})], **criteria())
    assert result["pair_count"] == 2
    assert {row["status"] for row in result["pairs"]} == {"missing_legacy", "missing_planned"}
    assert result["acceptance"]["status"] == "not_assessed"


@pytest.mark.parametrize("mode", ["legacy", "planned"])
def test_duplicate_same_mode_cannot_be_overwritten(mode):
    with pytest.raises(ValueError, match=f"duplicate {mode}"):
        qc.summarize_quality_cost([record(mode), record(mode, quality=0.)])


@pytest.mark.parametrize("field,value", [("task", "classification"),
                                         ("greater_is_better", True), ("memory_scope", "process_rss")])
def test_incompatible_pairs_cannot_pass_policy(field, value):
    result = pair([record(), record("planned", **{field: value})], **criteria())
    assert result["status"] == "incompatible"
    assert result["issues"] == [f"{field}_mismatch"]
    assert result["acceptance"]["status"] == "not_assessed"
    if field in ("task", "greater_is_better"):
        assert result["quality"]["planned_improvement"] is None


def test_memory_scope_is_explicit_and_tracemalloc_is_not_process_memory():
    with pytest.raises(ValueError, match="memory_scope"):
        qc.validate_run_record(record(peak_memory_bytes=0))
    with pytest.raises(ValueError, match="memory_scope"):
        qc.validate_run_record(record(peak_memory_bytes=0, memory_scope="not_measured"))
    rows = [record(peak_memory_bytes=100, memory_scope="python_tracemalloc"),
            record("planned", peak_memory_bytes=50, memory_scope="process_rss")]
    result = pair(rows, **criteria(max_memory_ratio=1.))
    assert result["costs"]["peak_memory_bytes"]["legacy"] == 100
    assert result["costs"]["peak_memory_bytes"]["ratio"] is None
    assert result["costs"]["peak_memory_bytes"]["difference"] is None
    assert result["costs"]["peak_memory_bytes"]["assessment"] == "not_assessed"
    assert result["costs"]["peak_memory_bytes"]["ratio_reason"] == "memory_scope_mismatch"
    rows[1]["memory_scope"] = "python_tracemalloc"
    result = qc.summarize_quality_cost(rows, **criteria(max_memory_ratio=1.))
    assert result["acceptance"]["status"] == "accepted"
    assert result["pairs"][0]["costs"]["peak_memory_bytes"]["ratio"] == .5
    assert "not process, native or CUDA" in result["memory_note"]


@pytest.mark.parametrize("new", [0., 1.])
def test_zero_baseline_is_undefined_not_a_free_speedup(new):
    rows = [record(train_seconds=-0.), record("planned", train_seconds=new)]
    result = pair(rows, **criteria())
    assert result["costs"]["train_seconds"]["difference"] == new
    assert result["costs"]["train_seconds"]["ratio"] is None
    assert result["costs"]["train_seconds"]["ratio_reason"] == "zero_baseline"
    assert result["acceptance"]["status"] == "not_assessed"
    # An explicit absolute-cost policy needs no ratio, including at zero.
    assert pair(rows, **criteria(max_train_seconds=new))["acceptance"]["status"] == "accepted"


def test_overflow_outputs_remain_json_safe_and_cannot_pass_undefined_checks():
    rows = [record(quality=1e308, train_seconds=5e-324),
            record("planned", quality=-1e308, train_seconds=1e308)]
    result = qc.summarize_quality_cost(rows, **criteria())
    assert result["pairs"][0]["quality"]["planned_improvement"] is None
    assert result["pairs"][0]["quality"]["reason"] == "arithmetic_overflow"
    assert result["pairs"][0]["costs"]["train_seconds"]["ratio"] is None
    assert result["acceptance"]["status"] == "not_assessed"
    json.dumps(result, allow_nan=False)


def test_ratio_underflow_does_not_fabricate_a_zero_cost():
    rows = [record(train_seconds=1e308), record("planned", train_seconds=5e-324)]
    result = pair(rows, **criteria(max_train_ratio=0))
    assert result["costs"]["train_seconds"]["ratio"] is None
    assert result["costs"]["train_seconds"]["ratio_reason"] == "arithmetic_underflow"
    assert result["acceptance"]["status"] == "not_assessed"
    rows[1]["train_seconds"] = 0.
    assert pair(rows, **criteria(max_train_ratio=0))["acceptance"]["status"] == "accepted"


def test_large_memory_integers_remain_exact_in_differences_and_absolute_limits():
    memory = 10 ** 400
    rows = [record(peak_memory_bytes=memory, memory_scope="process_rss"),
            record("planned", peak_memory_bytes=memory + 1, memory_scope="process_rss")]
    result = pair(rows, **criteria(max_peak_memory_bytes=memory + 1))
    assert result["costs"]["peak_memory_bytes"]["difference"] == 1
    assert result["acceptance"]["status"] == "accepted"
    json.dumps(result, allow_nan=False)
    # The displayed float ratio rounds to 1, but the exact constraint must fail.
    result = pair(rows, **criteria(max_memory_ratio=1))
    assert result["costs"]["peak_memory_bytes"]["ratio"] == 1.
    assert result["acceptance"]["status"] == "rejected"


def test_mixed_numeric_operands_do_not_lose_quality_or_cost_difference():
    old = 10 ** 30
    new = float(old)
    exact_difference = int(new) - old
    assert exact_difference > 0
    result = pair([record(quality=old, train_seconds=old),
                   record("planned", quality=new, train_seconds=new)],
                  quality_tolerance=0, cost_constraints={"max_train_ratio": 1})
    assert result["quality"]["planned_improvement"] == -exact_difference
    assert result["costs"]["train_seconds"]["difference"] == exact_difference
    assert result["acceptance"]["checks"] == {"quality_tolerance": False, "max_train_ratio": False}
    assert result["acceptance"]["status"] == "rejected"


def test_quality_constraint_is_not_relaxed_by_rounded_subtraction():
    # float(0.1) - float(1.1) rounds to -1.0, but the represented decline is larger.
    result = pair([record(quality=1.1, greater_is_better=True),
                   record("planned", quality=.1, greater_is_better=True)],
                  quality_tolerance=1., cost_constraints={"max_train_ratio": 1})
    assert result["quality"]["planned_improvement"] == -1.
    assert result["acceptance"]["checks"]["quality_tolerance"] is False
    assert result["acceptance"]["status"] == "rejected"


@pytest.mark.parametrize("policy", [
    {"quality_tolerance": .1}, {"cost_constraints": {"max_train_ratio": 1.}},
    criteria(unknown=1), criteria(max_train_ratio=True), criteria(max_train_ratio=-1),
    criteria(max_train_ratio=float("nan")), criteria(max_peak_memory_bytes=1.5),
    {"quality_tolerance": True, "cost_constraints": {"max_train_ratio": 1}},
    {"quality_tolerance": -1, "cost_constraints": {"max_train_ratio": 1}},
    {"quality_tolerance": .1, "cost_constraints": {}},
])
def test_explicit_policy_must_be_complete_and_valid(policy):
    with pytest.raises(ValueError):
        qc.summarize_quality_cost([], **policy)


def test_acceptance_uses_quality_and_requested_costs_never_graph_counts():
    rows = [record(), record("planned", quality=1.5, train_seconds=2., unique_graphs=0)]
    result = qc.summarize_quality_cost(rows, **criteria(max_train_ratio=.5, max_predict_seconds=2.))
    assert result["acceptance"]["status"] == "accepted"
    rows[1].update(quality=3., unique_graphs=10 ** 6)
    result = qc.summarize_quality_cost(rows, **criteria())
    assert result["acceptance"]["status"] == "rejected"
    assert result["pairs"][0]["acceptance"]["checks"]["quality_tolerance"] is False
    rows[1].update(quality=1.5, train_seconds=8.)
    assert qc.summarize_quality_cost(rows, **criteria())["acceptance"]["status"] == "rejected"
    assert pair(cost_constraints={"max_memory_ratio": 1.}, quality_tolerance=.1)[
        "acceptance"]["status"] == "not_assessed"


def test_empty_and_partially_complete_evidence_cannot_be_accepted():
    assert qc.summarize_quality_cost([], **criteria())["acceptance"]["status"] == "not_assessed"
    rows = [record(), record("planned"), record(dataset="unpaired")]
    result = qc.summarize_quality_cost(rows, **criteria())
    assert result["status_counts"]["succeeded"] == 1
    assert result["status_counts"]["missing_planned"] == 1
    assert result["record_count"] == 3
    assert result["acceptance"]["status"] == "not_assessed"


def test_permutation_invariance_without_cross_dataset_aggregation():
    rows = [record(), record("planned"), record(dataset="other", metric="mae")]
    expected = qc.summarize_quality_cost(rows)
    for shuffled in permutations(rows):
        assert qc.summarize_quality_cost(list(shuffled)) == expected
    assert len(expected["pairs"]) == 2


@given(old=st.integers(-100000, 100000), new=st.integers(-100000, 100000),
       higher=st.booleans(), baseline=st.integers(1, 100000), planned=st.integers(0, 100000))
def test_generated_pair_matches_independent_arithmetic_and_roundtrips(old, new, higher, baseline, planned):
    rows = [record(quality=old, greater_is_better=higher, train_seconds=baseline),
            record("planned", quality=new, greater_is_better=higher, train_seconds=planned)]
    result = qc.summarize_quality_cost(rows)
    delta = result["pairs"][0]["quality"]["planned_improvement"]
    assert delta == (new - old if higher else old - new)
    assert result["pairs"][0]["costs"]["train_seconds"]["ratio"] == planned / baseline
    assert qc.validate_run_record(qc.validate_run_record(rows[0])) == rows[0]
    assert qc.summarize_quality_cost(json.loads(json.dumps(rows))) == result


@given(tolerance=st.integers(0, 100), degradation=st.integers(0, 100))
def test_stricter_quality_policy_never_adds_acceptance(tolerance, degradation):
    rows = [record(quality=0), record("planned", quality=degradation)]
    loose = qc.summarize_quality_cost(rows, quality_tolerance=tolerance,
                                      cost_constraints={"max_train_ratio": 1.})
    strict = qc.summarize_quality_cost(rows, quality_tolerance=0,
                                       cost_constraints={"max_train_ratio": 1.})
    assert (strict["acceptance"]["status"] == "accepted") <= (loose["acceptance"]["status"] == "accepted")
    assert (loose["acceptance"]["status"] == "accepted") == (degradation <= tolerance)


def test_cli_writes_json_markdown_without_loading_ml_packages(tmp_path):
    source = tmp_path / "records.json"
    source.write_text(json.dumps({"schema_version": qc.SCHEMA_VERSION,
                                 "records": [record(), record("planned")]}), encoding="utf-8")
    output, markdown = tmp_path / "report.json", tmp_path / "report.md"
    completed = subprocess.run([sys.executable, "-I", str(MODULE_PATH), str(source),
                                "--output-json", str(output), "--output-markdown", str(markdown)],
                               capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["acceptance"]["status"] == "not_assessed"
    text = markdown.read_text(encoding="utf-8")
    assert "not_assessed" in text and "Unique graph counts are not acceptance" in text
    assert "predict_seconds" in text and "peak_memory_bytes" in text
    assert "python_tracemalloc" in text


def test_cli_explicit_criteria_and_invalid_envelope(tmp_path, capsys):
    source, policy = tmp_path / "records.json", tmp_path / "criteria.json"
    source.write_text(json.dumps({"schema_version": qc.SCHEMA_VERSION,
                                 "records": [record(), record("planned")]}), encoding="utf-8")
    policy.write_text(json.dumps(criteria()), encoding="utf-8")
    assert qc.main([str(source), "--criteria", str(policy)]) == 0
    assert json.loads(capsys.readouterr().out)["acceptance"]["status"] == "accepted"
    for invalid in ({"schema_version": "old", "records": []},
                    {"schema_version": qc.SCHEMA_VERSION, "records": [record(quality=float("nan"))]}):
        source.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(SystemExit) as error:
            qc.main([str(source)])
        assert error.value.code == 2


def test_cli_rejects_duplicate_json_fields_before_pairing(tmp_path):
    source = tmp_path / "records.json"
    source.write_text('{"schema_version":"' + qc.SCHEMA_VERSION + '","records":[],"records":[]}',
                      encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        qc.main([str(source)])
    assert error.value.code == 2


@pytest.mark.parametrize("constraint,field,measured", [
    ("max_train_seconds", "train_seconds", 2.),
    ("max_predict_seconds", "predict_seconds", 2.),
    ("max_peak_memory_bytes", "peak_memory_bytes", 2),
    ("max_train_ratio", "train_seconds", 8.),
    ("max_predict_ratio", "predict_seconds", 4.),
    ("max_memory_ratio", "peak_memory_bytes", 4),
])
def test_all_cost_constraints_are_inclusive_and_enforced(constraint, field, measured):
    rows = [record(peak_memory_bytes=2, memory_scope="python_tracemalloc"),
            record("planned", peak_memory_bytes=2, memory_scope="python_tracemalloc")]
    rows[1][field] = measured
    assert pair(rows, **criteria(**{constraint: 2}))["acceptance"]["status"] == "accepted"
    assert pair(rows, **criteria(**{constraint: 1}))["acceptance"]["status"] == "rejected"


def test_markdown_escapes_user_labels_and_keeps_failed_pairs_visible():
    result = qc.summarize_quality_cost([record(scenario="bad|label\nnext", status="failed",
                                               quality=None, predict_seconds=None)])
    text = qc.render_quality_cost_markdown(result)
    assert "bad\\|label next" in text
    assert "missing_planned" in text and "failed" in text and "null" in text
