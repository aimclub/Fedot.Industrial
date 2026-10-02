"""Compatibility facade around the pure Industrial graph validator."""

from __future__ import annotations

from fedot_ind.core.optimizer.graph_validation.adapters import project_runtime_graph
from fedot_ind.core.optimizer.graph_validation.contracts import (
    ValidationIssue,
    ValidationIssueCode,
    ValidationReport,
)
from fedot_ind.core.optimizer.graph_validation.validation import validate_graph


EXPECTED_GRAPH_ADAPTATION_ERRORS = (
    AttributeError,
    IndexError,
    KeyError,
    TypeError,
    ValueError,
)


class IndustrialGraphVerifier:
    """Expose the historical bool verifier API and retain a complete report."""

    def __init__(
            self,
            *,
            adapter: object | None,
            task_type: object | None,
            requested_device: object | None = None,
            require_serializable: bool = False,
            legacy_verifier: object | None = None,
    ) -> None:
        self.adapter = adapter
        self.task_type = task_type
        self.requested_device = requested_device
        self.require_serializable = require_serializable
        self.legacy_verifier = legacy_verifier
        self.last_report: ValidationReport | None = None

    def __call__(self, graph: object) -> bool:
        return self.verify(graph)

    def verify(self, graph: object) -> bool:
        """Return whether the graph is valid and retain the report in ``last_report``.

        Use the error handling and optional legacy check of ``verify_with_report``.
        """
        return self.verify_with_report(graph).is_valid

    def verify_with_report(self, graph: object) -> ValidationReport:
        """Validate a runtime graph and retain the returned report in ``last_report``.

        Run the optional legacy verifier only after the pure rules accept the
        graph. AttributeError, IndexError, KeyError, TypeError, and ValueError from
        graph projection or legacy verification become report issues. Other
        exceptions, including OSError, propagate; pure-rule exceptions also
        propagate.
        """
        try:
            runtime_graph, spec = project_runtime_graph(
                graph,
                adapter=self.adapter,
                task_type=self.task_type,
                requested_device=self.requested_device,
                require_serializable=self.require_serializable,
            )
        except EXPECTED_GRAPH_ADAPTATION_ERRORS as error:
            report = ValidationReport(
                graph_id=str(getattr(graph, "uid", type(graph).__name__)),
                issues=(ValidationIssue.create(
                    ValidationIssueCode.GRAPH_ADAPTATION_FAILED,
                    "Runtime graph could not be converted into a validation specification.",
                    context={"error_type": type(error).__name__, "message": str(error)},
                ),),
            )
            self.last_report = report
            return report

        report = validate_graph(spec)
        if report.is_valid and self.legacy_verifier is not None:
            try:
                legacy_valid = bool(self.legacy_verifier(runtime_graph))
            except EXPECTED_GRAPH_ADAPTATION_ERRORS as error:
                report = report.append(ValidationIssue.create(
                    ValidationIssueCode.LEGACY_VERIFIER_FAILED,
                    "The compatibility verifier failed while checking the graph.",
                    context={"error_type": type(error).__name__, "message": str(error)},
                ))
            else:
                if not legacy_valid:
                    report = report.append(ValidationIssue.create(
                        ValidationIssueCode.LEGACY_VERIFIER_REJECTED,
                        "The compatibility verifier rejected a graph accepted by pure rules.",
                    ))
        self.last_report = report
        return report
