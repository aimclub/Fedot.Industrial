"""One shared coordinator-side interpretation of worker result metadata."""

from fedot_ind.core.optimizer.domain import EvaluationFailed, EvaluationReused, evaluation_outcome_from_metadata
from fedot_ind.core.optimizer.observability import EvaluationStatus, graph_identity


def record_evaluation_results(recorder, results):
    if recorder is None:
        return
    for result in results:
        if result is None:
            continue
        outcome = evaluation_outcome_from_metadata(dict(result.metadata or {}))
        failure = outcome.failure if isinstance(outcome, EvaluationFailed) else None
        status = (EvaluationStatus.FAILED if failure is not None else
                  EvaluationStatus.REUSED if isinstance(outcome, EvaluationReused) else EvaluationStatus.SUCCEEDED)
        recorder.record_evaluation(
            individual_id=str(result.uid_of_individual), graph_id=graph_identity(result.graph), status=status,
            duration_seconds=outcome.duration_seconds, error_type=failure.error_type if failure else None,
            error_message=failure.message if failure else None,
        )
