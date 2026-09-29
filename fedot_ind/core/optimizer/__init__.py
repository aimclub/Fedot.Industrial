"""Industrial evolutionary optimisation contracts."""

from fedot_ind.core.optimizer.observability import (
    CandidateRejectionReason,
    CandidateStatus,
    EvaluationStatus,
    EvolutionDiagnosticsRecorder,
    EvolutionDiagnosticsSnapshot,
    EvolutionSummary,
    build_evolution_summary,
    write_evolution_diagnostics,
)

__all__ = [
    "CandidateRejectionReason",
    "CandidateStatus",
    "EvaluationStatus",
    "EvolutionDiagnosticsRecorder",
    "EvolutionDiagnosticsSnapshot",
    "EvolutionSummary",
    "build_evolution_summary",
    "write_evolution_diagnostics",
]
