"""Search termination is a control transition, not another failed offspring."""


class SearchBudgetExhausted(RuntimeError):
    """A search with completed evaluations has no time for another batch."""


def search_deadline_reached(timeout_seconds, elapsed_seconds):
    return timeout_seconds is not None and elapsed_seconds >= timeout_seconds
