"""Proof of local process termination for a blocked B/D receipt."""
from dataclasses import asdict
import json
from math import isfinite
from socket import gethostname

import psutil

from benchmark.industrial.experiments.mutation_quality.execution.runner import ProtocolViolation
from benchmark.industrial.experiments.mutation_quality.preflight import file_digest
from fedot_ind.core.optimizer.evaluation.dask_executor import _worker_process_alive


def failure_record(error):
    owner = psutil.Process()
    return dict(schema_version='industrial_bd_cleanup_failure@1', host=gethostname(), boot_time=psutil.boot_time(),
                owner=(owner.pid, owner.create_time()), error=str(error),
                shutdown=None if error.shutdown is None else asdict(error.shutdown))


def _identity(raw):
    if (type(raw) is not list or len(raw) != 2 or type(raw[0]) is not int or raw[0] <= 0
            or type(raw[1]) not in (int, float) or not isfinite(raw[1]) or raw[1] <= 0):
        raise ProtocolViolation('Invalid process identity in Dask cleanup evidence')
    return tuple(raw)


def confirmed_cleanup(directory, receipt_path, row):
    """Do not override other infrastructure failures or stop another process."""
    member = 'worker_failure.json'
    if row['exit_code'] != 2 or member not in row['files']:
        raise ProtocolViolation('The blocked trial has no recorded Dask cleanup failure')
    try:
        failure = json.loads((directory / member).read_text('utf-8'))
        boot_time = failure['boot_time']
        if (failure['schema_version'] != 'industrial_bd_cleanup_failure@1' or failure['host'] != gethostname()
                or type(boot_time) not in (int, float) or not isfinite(boot_time)
                or abs(psutil.boot_time() - boot_time) > 2):
            raise ProtocolViolation('Dask cleanup must be checked on the same host and boot')
        shutdown = failure['shutdown']
        if (not isinstance(shutdown, dict) or shutdown.get('cluster_closed') is not True
                or shutdown.get('inventory_complete') is not True or type(shutdown.get('workers')) is not list):
            raise ProtocolViolation('Cluster closure or complete worker inventory was not confirmed')
        identities = [_identity(failure['owner']), *(_identity(item) for item in shutdown['workers'])]
        if any(_worker_process_alive(identity) for identity in identities):
            raise ProtocolViolation('A process owned by the blocked trial is still alive')
    except (OSError, ValueError, TypeError, KeyError, psutil.Error) as error:
        raise ProtocolViolation('Dask cleanup evidence could not be verified') from error
    return dict(schema_version='industrial_bd_cleanup_confirmation@1', receipt_sha256=file_digest(receipt_path),
                host=failure['host'], boot_time=boot_time, stopped_processes=[list(item) for item in identities])
