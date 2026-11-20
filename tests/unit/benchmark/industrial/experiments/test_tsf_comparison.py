import tracemalloc

import numpy as np
import pytest

from benchmark.industrial.experiments.tsf_comparison import (
    ForecastComparisonPlan, SourceFingerprint, comparison_provenance, comparison_report, forecast_rows, measure,
    panel_holdout,
)


def test_forecast_windows_exclude_future_and_isolate_methods():
    source = np.arange(8.)
    histories = []

    def mutates(history, horizon):
        origin = len(history)
        history[:] = -1
        return np.arange(origin, origin + horizon)

    def clean(history, horizon):
        histories.append(history)
        return np.repeat(-1., horizon)

    rows = list(forecast_rows(source, {"mutates": mutates, "clean": clean}, ForecastComparisonPlan((3, 6), 2)))
    assert [r["rmse"] for r in rows if r["model"] == "mutates"] == [0., 0.]
    np.testing.assert_array_equal(histories[0], [0., 1., 2.])
    np.testing.assert_array_equal(histories[1], np.arange(6.))
    np.testing.assert_array_equal(source, np.arange(8.))
    with pytest.raises(ValueError, match="TRAIN boundary"):
        list(forecast_rows(source, {"mutates": mutates}, ForecastComparisonPlan((3, 7), 2)))
    with pytest.raises(ValueError, match="unique"):
        ForecastComparisonPlan((3, 3), 2)


def test_group_holdout_is_deterministic_and_preserves_record_ownership():
    labels = np.repeat((0, 1), 12)
    groups = np.repeat(np.arange(8), 3)
    fit, evaluate = panel_holdout(labels, seed=42, stratified=True, groups=groups)
    fit_next, evaluate_next = panel_holdout(labels, seed=42, stratified=True, groups=groups)
    np.testing.assert_array_equal(fit, fit_next)
    np.testing.assert_array_equal(evaluate, evaluate_next)
    np.testing.assert_array_equal(np.sort(np.r_[fit, evaluate]), np.arange(len(labels)))
    assert set(groups[fit]).isdisjoint(groups[evaluate])
    for part in (fit, evaluate):
        assert set(labels[part]) == {0, 1}
    with pytest.raises(ValueError, match="group identifier"):
        panel_holdout(labels, seed=42, stratified=True, groups=groups[:-1])


def test_provenance_detects_input_changes_and_records_unavailable_metadata(tmp_path):
    groups = tmp_path / "groups"
    groups.write_bytes(b"one\ntwo\n")
    first = SourceFingerprint.file(groups)
    groups.write_bytes(b"two\none\n")
    second = SourceFingerprint.file(groups)
    assert first.name == second.name and first.sha256 != second.sha256
    assert SourceFingerprint.arrays("x", [1, 2]) != SourceFingerprint.arrays("x", [[1, 2]])
    assert SourceFingerprint.arrays("x", [1, 2]) != SourceFingerprint.arrays("x", [1., 2.])
    provenance = comparison_provenance({"groups": second}, ("numpy", "not-a-tsf-package"), repository=tmp_path)
    report = comparison_report("early", provenance=provenance, protocol={"partition": "TRAIN"},
                               results={"rows": [], "schema_version": "invalid"})
    assert report["schema_version"] == "industrial_tsf_comparison@1"
    assert report["provenance"]["packages"]["numpy"] == np.__version__
    assert report["provenance"]["packages"]["not-a-tsf-package"] is None
    assert report["provenance"]["sources"]["groups"]["sha256"] == second.sha256


def test_allocation_measurement_releases_only_its_own_trace():
    def fail():
        raise ValueError("operation failed")

    with pytest.raises(ValueError, match="operation failed"):
        measure(fail, trace_memory=True)
    assert not tracemalloc.is_tracing()
    tracemalloc.start()
    try:
        with pytest.raises(RuntimeError, match="active allocation trace"):
            measure(fail, trace_memory=True)
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()
