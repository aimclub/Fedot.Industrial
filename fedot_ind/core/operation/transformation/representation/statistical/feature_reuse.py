"""Bounded, explicitly owned reuse of deterministic statistical features.

This stores values, never fitted operations. It is not the legacy disk cache.
Only the audited QuantileExtractor opts in; other transforms stay unchanged.
"""

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock
from typing import Callable, Iterator

import numpy as np


@dataclass(frozen=True)
class ReuseStatistics:
    hits: int
    misses: int
    bypasses: int
    retained_bytes: int
    peak_bytes: int


class FeatureReuseSession:
    """Own private snapshots and discard them even when the caller fails."""

    def __init__(self, max_bytes: int, max_entries: int) -> None:
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError('max_bytes must be a nonnegative integer')
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError('max_entries must be a positive integer')
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self._entries = OrderedDict()
        self._hits = self._misses = self._bypasses = self._bytes = self._peak = 0
        self._closed = False
        self._lock = RLock()

    @property
    def statistics(self) -> ReuseStatistics:
        with self._lock:
            return ReuseStatistics(self._hits, self._misses, self._bypasses, self._bytes, self._peak)

    def evaluate(self, features: np.ndarray, settings: tuple, compute: Callable[[], np.ndarray]) -> np.ndarray:
        # A copied context may outlive a timed-out owner. A closed session cannot
        # retain late worker results. Never hold the lock during native compute:
        # closing a timed-out scope must not wait for that computation to finish.
        with self._lock:
            bypass = (self._closed or not self.max_bytes or type(features) is not np.ndarray
                      or not features.flags.c_contiguous or features.dtype.kind not in 'biufc'
                      or features.nbytes > self.max_bytes)
            if bypass:
                self._bypasses += 1
        if bypass:
            return compute()
        key = (settings, features.dtype.str, features.shape,
               sha256(features.tobytes(order='C')).digest())
        with self._lock:
            if key in self._entries:
                self._hits += 1
                value = self._entries.pop(key)
                self._entries[key] = value
                return value.copy()
            self._misses += 1
        value = compute()
        with self._lock:
            if self._closed or not isinstance(value, np.ndarray) or value.nbytes > self.max_bytes:
                return value
            if key in self._entries:
                self._bytes -= self._entries.pop(key).nbytes
            while self._entries and (self._bytes + value.nbytes > self.max_bytes
                                     or len(self._entries) >= self.max_entries):
                self._bytes -= self._entries.popitem(last=False)[1].nbytes
            snapshot = value.copy()
            snapshot.flags.writeable = False
            self._entries[key] = snapshot
            self._bytes += snapshot.nbytes
            self._peak = max(self._peak, self._bytes)
            return value

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._entries.clear()
            self._bytes = 0


_SESSION: ContextVar[FeatureReuseSession | None] = ContextVar('industrial_feature_reuse', default=None)


@contextmanager
def feature_reuse_scope(*, max_bytes: int = 32 * 1024 * 1024,
                        max_entries: int = 128) -> Iterator[FeatureReuseSession]:
    """Create an isolated scope; nested calls never borrow the owner's entries."""
    session = FeatureReuseSession(max_bytes, max_entries)
    token = _SESSION.set(session)
    try:
        yield session
    finally:
        session.close()
        _SESSION.reset(token)


def reuse_statistical_features(features: np.ndarray, settings: tuple,
                               compute: Callable[[], np.ndarray]) -> np.ndarray:
    session = _SESSION.get()
    return compute() if session is None else session.evaluate(features, settings, compute)


def run_with_feature_reuse(operation: Callable, *args, **kwargs):
    """Run one fit/predict with fresh storage, including exception cleanup."""
    with feature_reuse_scope():
        return operation(*args, **kwargs)
