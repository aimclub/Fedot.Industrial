from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from fedot_ind.core.kernel_learning.contracts import KernelBundle


@dataclass(frozen=True)
class KernelCachePolicy:
    enabled: bool = False
    namespace: str = "kernel_learning"


@dataclass(frozen=True)
class KernelCacheKey:
    namespace: str
    generator_name: str
    kernel_policy_hash: str
    data_fingerprint: str

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (
            self.namespace,
            self.generator_name,
            self.kernel_policy_hash,
            self.data_fingerprint,
        )


class InMemoryKernelCache:
    def __init__(self):
        self._storage: dict[tuple[str, str, str, str], KernelBundle] = {}

    def get(self, key: KernelCacheKey) -> KernelBundle | None:
        return self._storage.get(key.as_tuple())

    def put(self, key: KernelCacheKey, bundle: KernelBundle) -> KernelBundle:
        self._storage[key.as_tuple()] = bundle
        return bundle

    def clear(self) -> None:
        self._storage.clear()

    @property
    def size(self) -> int:
        return len(self._storage)


def fingerprint_array(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    if array.dtype == object:
        digest.update("|".join(map(str, array.reshape(-1))).encode("utf-8"))
    else:
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def fingerprint_mapping(values: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(values):
        digest.update(str(key).encode("utf-8"))
        digest.update(str(values[key]).encode("utf-8"))
    return digest.hexdigest()
