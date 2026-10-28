"""Shared Python/NumPy RNG ownership without importing an optimizer runtime."""

from contextlib import contextmanager
import random
from threading import RLock

import numpy as np


_RANDOM_STATE_LOCK = RLock()


@contextmanager
def operator_random_scope(seed: int | None):
    """Serialize participating scopes and restore seeded ambient RNGs on failure.

    Coordinator operators and local evaluation share this reentrant lock.
    No Torch or estimator-owned RNG state is changed. Unseeded scopes retain
    their ambient draws but cannot overlap seeded ones. Unrelated code ignoring
    this lock is outside the reproducibility contract.
    """
    with _RANDOM_STATE_LOCK:
        if seed is None:
            yield
            return
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
