"""One parent-launch deadline, including imports and executor construction."""
from datetime import datetime, timedelta


def absolute_search_timer(launched_at, deadline):
    from golem.core.optimisers.timer import OptimisationTimer

    if deadline <= launched_at:
        raise ValueError('Search deadline must follow launch')

    class LaunchTimer(OptimisationTimer):
        def __enter__(self):
            self.start = datetime.fromtimestamp(launched_at)
            return self

    return LaunchTimer(timeout=timedelta(seconds=deadline - launched_at))


def search_complete(evolved, required):
    """Time-limited and generation-limited searches have the same evidence bar."""
    return evolved >= required
