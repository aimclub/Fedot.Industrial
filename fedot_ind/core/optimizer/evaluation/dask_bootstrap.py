"""Worker-local verification; imports never create a Dask client or cluster."""

import json
import os
from pathlib import Path

import cloudpickle

from distributed import WorkerPlugin, get_worker

from fedot_ind.integration.fedot.extensions.catalog import (
    load_industrial_extension_catalog, parse_industrial_extension_catalog,
)


class IndustrialWorkerBootstrap(WorkerPlugin):
    """Verify a delivered data-only manifest against installed worker resources."""

    name = "industrial-evaluation-bootstrap"
    idempotent = True

    def __init__(self, catalog_json):
        self.catalog_json = catalog_json

    def setup(self, worker):
        expected = parse_industrial_extension_catalog(json.loads(self.catalog_json))
        installed = load_industrial_extension_catalog()
        if installed.digest != expected.digest:
            raise ValueError("Worker Industrial catalog differs from the delivered manifest")
        if worker.state.nthreads != 1:
            raise ValueError("Industrial evaluation requires one task thread per worker")
        if worker.memory_manager.memory_limit <= 0:
            raise ValueError("Industrial evaluation requires a positive worker memory budget")
        worker._industrial_evaluation_manifest = expected.digest


def worker_resources(dask_worker=None):
    worker = dask_worker if dask_worker is not None else get_worker()
    return {"digest": getattr(worker, "_industrial_evaluation_manifest", None),
            "pid": os.getpid(), "threads": worker.state.nthreads,
            "memory_limit": worker.memory_manager.memory_limit}


def evaluate_on_worker(task, payload, evaluate, digest, result_path):
    from .executors import _invoke

    if getattr(get_worker(), "_industrial_evaluation_manifest", None) != digest:
        raise ValueError("Worker did not verify the Industrial evaluation manifest")
    path = Path(result_path)
    # Dask can recompute lost receipts even with failure retries disabled. An
    # already persisted evaluation is returned without invoking the model again.
    if not path.exists():
        result = _invoke(task, payload, evaluate)
        temporary = path.with_suffix(".partial")
        temporary.write_bytes(cloudpickle.dumps(result))
        temporary.replace(path)
    return str(path)
