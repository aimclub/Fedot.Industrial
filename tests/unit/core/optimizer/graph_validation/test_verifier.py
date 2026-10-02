"""Unit coverage for failure handling at the runtime graph boundary."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fedot_ind.core.optimizer.graph_validation import (
    GraphNodeSpec,
    GraphSpec,
    IndustrialGraphVerifier,
    OperationCapabilities,
    OperationKind,
    ValidationIssueCode,
)
from fedot_ind.core.optimizer.graph_validation import verifier as verifier_module


@pytest.fixture
def projected_graph(monkeypatch):
    runtime = object()
    spec = GraphSpec("valid", nodes=(GraphNodeSpec(
        "root", "model", capabilities=OperationCapabilities(
            name="model", kind=OperationKind.MODEL,
            input_data_types=("tabular",), output_data_types=("tabular",),
            task_types=("regression",),
        ),
    ),), task_type="regression")
    project = Mock(return_value=(runtime, spec))
    monkeypatch.setattr(verifier_module, "project_runtime_graph", project)
    return project, runtime


@pytest.mark.parametrize("boundary", ["adaptation", "legacy"])
@pytest.mark.parametrize("error_type", [AttributeError, IndexError, KeyError, TypeError, ValueError])
def test_expected_boundary_errors_retain_cause_and_replace_previous_report(
        projected_graph, boundary, error_type):
    project, runtime = projected_graph
    legacy = Mock(return_value=True)
    verifier = IndustrialGraphVerifier(adapter=None, task_type="regression", legacy_verifier=legacy)
    graph = SimpleNamespace(uid="candidate")
    assert verifier.verify_with_report(graph).is_valid
    legacy.reset_mock()
    failure = error_type("invalid graph")
    (project if boundary == "adaptation" else legacy).side_effect = failure

    report = verifier.verify_with_report(graph)

    expected_code = (ValidationIssueCode.GRAPH_ADAPTATION_FAILED if boundary == "adaptation"
                     else ValidationIssueCode.LEGACY_VERIFIER_FAILED)
    assert report.error_codes == (expected_code.value,)
    assert verifier.last_report is report
    assert report.issues[0].to_record()["context"] == {
        "error_type": error_type.__name__, "message": str(failure),
    }
    if boundary == "adaptation":
        assert report.graph_id == "candidate"
        legacy.assert_not_called()
    else:
        legacy.assert_called_once_with(runtime)


@pytest.mark.parametrize("boundary", ["adaptation", "legacy"])
@pytest.mark.parametrize("error_type", [OSError, RuntimeError, MemoryError])
def test_infrastructure_errors_are_not_disguised_as_invalid_candidates(
        projected_graph, boundary, error_type):
    project, _ = projected_graph
    legacy = Mock(return_value=True)
    failure = error_type("runtime unavailable")
    (project if boundary == "adaptation" else legacy).side_effect = failure
    verifier = IndustrialGraphVerifier(adapter=None, task_type="regression", legacy_verifier=legacy)

    with pytest.raises(error_type) as error:
        verifier.verify_with_report(object())

    assert error.value is failure
    assert verifier.last_report is None


def test_successful_retry_replaces_previous_failure_report(projected_graph):
    _, runtime = projected_graph
    legacy = Mock(side_effect=[ValueError("bad metadata"), True])
    verifier = IndustrialGraphVerifier(adapter=None, task_type="regression", legacy_verifier=legacy)

    rejected = verifier.verify_with_report(object())
    accepted = verifier.verify_with_report(object())

    assert not rejected.is_valid
    assert accepted.is_valid
    assert accepted.issues == ()
    assert verifier.last_report is accepted
    assert legacy.call_count == 2
    legacy.assert_called_with(runtime)
