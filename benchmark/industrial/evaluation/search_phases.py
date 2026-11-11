"""Pure phase accounting: coordinator wall time and worker sums stay separate."""
import math


def summarize_search_phases(*, setup_seconds, search_seconds, batches, sessions, lifecycle,
                            refit_seconds, predict_seconds):
    def total(rows, key):
        return sum(float(row[key]) for row in rows if row.get(key) is not None)

    preparation = total(batches, 'preparation_seconds')
    execution = total(batches, 'execution_seconds')
    application = total(batches, 'application_seconds')
    startup = total(sessions, 'startup_seconds')
    cleanup = total(sessions, 'cleanup_seconds')
    # Search-owned cleanup is outside batch execution; batch cleanup is inside it.
    outside_cleanup = cleanup if lifecycle == 'search' else 0.
    other = max(0., search_seconds - preparation - execution - application - outside_cleanup)
    result = dict(setup_seconds=setup_seconds, search_seconds=search_seconds,
                  batch_preparation_seconds=preparation, batch_execution_seconds=execution,
                  result_application_seconds=application, coordinator_other_seconds=other,
                  session_startup_seconds=startup, session_cleanup_seconds=cleanup,
                  refit_seconds=refit_seconds, predict_seconds=predict_seconds)
    if any(not math.isfinite(v) or v < 0 for v in result.values() if v is not None):
        raise ValueError('Phase durations must be finite and nonnegative')
    return result
