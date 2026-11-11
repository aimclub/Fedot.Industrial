"""Worker-local verification; imports never create a Dask client or cluster."""

import json
import os
from importlib import import_module
from pathlib import Path
from time import monotonic

import cloudpickle
import psutil

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
        worker._industrial_evaluation_runtime = prepare_worker_runtime()
        worker._industrial_evaluation_manifest = expected.digest


def worker_resources(dask_worker=None):
    worker = dask_worker if dask_worker is not None else get_worker()
    return {"digest": getattr(worker, "_industrial_evaluation_manifest", None),
            "pid": os.getpid(), "create_time": psutil.Process().create_time(), "threads": worker.state.nthreads,
            "memory_limit": worker.memory_manager.memory_limit,
            "runtime": getattr(worker, '_industrial_evaluation_runtime', None)}


def prepare_worker_runtime():
    """Prepare shared imports/code before admission, never fit a user model.

    This is part of session startup and its total budget. Candidate deadlines
    still include submission, model execution and delivery; none are reset.
    """
    started = monotonic()
    import_module('fedot_ind.core.optimizer.evaluation.dispatcher')
    import numpy as np
    import torch
    from fedot_ind.core.operation.transformation.representation.statistical.hurst_batch import native_hurst_supported

    flags = (torch.get_num_threads(), torch.get_num_interop_threads(),
             torch.are_deterministic_algorithms_enabled())
    values = np.zeros(512, dtype=np.float32)
    warmed = native_hurst_supported(values)
    if warmed:
        import_module('fedot_ind.core.operation.transformation.representation.statistical.quantile_extractor')
        import_module('fedot_ind.core.operation.transformation.torch_backend.statistical.quantile_extractor')
        from fedot_ind.core.operation.transformation.representation.statistical.native_statistics import (
            prefixes, hjorth_m4,
        )
        cumulative = np.cumsum(values)
        prefixes(values, cumulative, cumulative / np.arange(1, len(values) + 1))
        hjorth_m4(values)
    after = (torch.get_num_threads(), torch.get_num_interop_threads(),
             torch.are_deterministic_algorithms_enabled())
    if flags != after:
        raise RuntimeError('Worker preparation changed Torch settings')
    return dict(seconds=monotonic() - started, native_statistics_warmed=warmed,
                model_fit_calls=0, torch_flags_preserved=True)


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
