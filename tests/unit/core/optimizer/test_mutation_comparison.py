import json
from pathlib import Path

import pytest

from benchmark.industrial.experiments.mutation_planning import (
    compare_mutations, summarize_attempts, write_comparison,
)


def test_summary_preserves_missing_denominators_instead_of_fake_zero_diversity():
    summary = summarize_attempts([])
    assert summary["valid_ratio"] is None
    assert summary["uniqueness_among_changed"] is None
    assert summary["unique_changed_graphs"] == 0


def test_summary_distinguishes_valid_unchanged_and_unique_changed():
    rows = [dict(valid=True, changed=changed, graph_id="same", operations=["ridge"],
                 status="accepted", elapsed_seconds=1.0) for changed in (True, True, False)]
    rows.append(dict(valid=False, changed=False, graph_id=None, operations=[], status="rejected", elapsed_seconds=1.0))
    summary = summarize_attempts(rows)
    assert summary["valid_ratio"] == .75
    assert summary["changed_valid_ratio"] == .5
    assert summary["uniqueness_among_changed"] == .5
    assert summary["operation_frequency"] == {"ridge": 2}


def _config():
    return {"schema_version": "industrial_mutation_comparison@1", "seeds": [42], "attempts_per_seed": 3,
            "max_candidates": 12, "max_depth": 5, "max_nodes": 8, "max_arity": 3,
            "scenarios": [{"name": "regression", "task": "regression",
                           "operations": ["scaling", "ridge", "dtreg"],
                           "initial_chain": ["scaling", "ridge"]}]}


def test_paired_measurement_repeats_choices_and_saves_inspectable_records(tmp_path):
    first, second = compare_mutations(_config()), compare_mutations(_config())
    for report in (first, second):
        for record in report["records"]:
            record.pop("elapsed_seconds")
    assert first["records"] == second["records"]
    assert len(first["records"]) == 6
    path = write_comparison(first, tmp_path)
    assert json.loads(path.read_text())["scope"] == "structural_proposals_without_training"
    assert Path(path).with_name("comparison.md").is_file()


@pytest.mark.parametrize("field,value", [("attempts_per_seed", True), ("max_candidates", 0), ("seeds", [])])
def test_measurement_rejects_invalid_limits(field, value):
    config = _config()
    config[field] = value
    with pytest.raises(ValueError):
        compare_mutations(config)
