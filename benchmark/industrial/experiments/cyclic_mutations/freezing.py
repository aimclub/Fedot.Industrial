"""New source seal referencing unchanged archived TRAIN data, never an old code seal."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import distributions
from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

from benchmark.industrial.experiments.mutation_quality.freeze import verify_freeze, seal, write_json
from benchmark.industrial.experiments.mutation_quality.preflight import file_digest
from benchmark.industrial.experiments.mutation_quality.execution.frozen_data import load_frozen_data

from .profile import parameter_profile, validate_training_bounds
from .provenance import platform_profile, source_inventory, verify_active_sources
from .study import policy_for


def freeze(base, base_seal, directory):
    import_module('fedot')  # Distribution visibility must match workers after FEDOT initialization.
    import torch
    base, directory = Path(base).resolve(), Path(directory).resolve()
    verify_freeze(base, check_current=False, expected_seal_sha256=base_seal)
    original = json.loads((base / 'lock.json').read_text('utf-8'))
    lock = deepcopy(original)
    lock['schema_version'] = 'industrial_bd_lock@2'
    lock['input_procedure_git_revision_context'] = original['git_revision_context']
    lock['git_revision_context'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    lock['execution']['full_runner_status'] = 'B/D runner; requires its own successful admission'
    lock['created_at'] = datetime.now(timezone.utc).isoformat()
    lock['base'] = dict(directory=str(base), seal=base_seal, lock_sha256=file_digest(base / 'lock.json'))
    lock['protocol']['arms'] = ['fixed', 'cyclic']
    lock['policies'] = {arm: policy_for(arm).to_record() for arm in lock['protocol']['arms']}
    tasks = {r['scenario']: r['task'] for r in original['jobs']}
    lock['jobs'] = [dict(scenario=name, task=tasks[name], arm=arm, seed=seed,
                         search_seconds=lock['protocol']['budget_seconds'],
                         minimum_generations=lock['protocol']['minimum_generations'],
                         full_process_seconds=1200)
                    for name in lock['datasets'] for seed in lock['protocol']['seeds']
                    for arm in lock['protocol']['arms']]
    admission = json.loads(Path(__file__).with_name('admission.json').read_text('utf-8'))
    scenarios = admission.get('scenarios') if type(admission) is dict else None
    if (type(admission) is not dict or set(admission) != {'schema_version', 'scenarios'}
            or admission['schema_version'] != 'industrial_bd_admission@1'
            or type(scenarios) is not list or len(scenarios) != 3
            or any(type(name) is not str or name not in tasks for name in scenarios)
            or len(set(scenarios)) != 3
            or {tasks[name] for name in scenarios} != {'classification', 'regression', 'ts_forecasting'}):
        raise ValueError('Admission requires three distinct known scenarios covering all tasks')
    lock['admission_scenarios'] = scenarios
    lock['scoring']['classification'] = 'balanced_accuracy'
    lock['scoring']['comparisons'] = ['cyclic-fixed']
    paths = source_inventory()
    directory.mkdir(parents=True, exist_ok=False)
    sources = {}
    with ZipFile(directory / 'sources.zip', 'x', compression=ZIP_DEFLATED) as archive:
        for name, path in paths.items():
            data = path.read_bytes()
            archive.writestr(f'sources/{name}', data)
            sources[name] = sha256(data).hexdigest()
    lock['sources'] = sources
    lock['platform_profile'] = platform_profile()
    checked = []
    for name in lock['datasets']:
        data = load_frozen_data(base, lock, name)
        validate_training_bounds(data, parameter_profile())
        if data.test is not None:
            raise ValueError('Search unexpectedly received control data')
        checked.append(dict(scenario=name, folds=len(data.folds), test_available=False))
    flags = dict(threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                 deterministic=torch.are_deterministic_algorithms_enabled())
    if flags != lock['execution']['torch_flags']:
        raise ValueError(f'Torch flags differ from the approved runtime: {flags}')
    lock['freeze_process_torch_flags'] = flags
    write_json(directory / 'data_preflight.json', dict(scenarios=checked, fit_calls=0, test_used=False))
    write_json(directory / 'environment.json', dict(python=sys.version, executable=sys.executable,
               packages={dist.metadata['Name']: dist.version for dist in distributions()}, torch_flags=flags))
    lock['environment_sha256'] = file_digest(directory / 'environment.json')
    verify_active_sources(sources)
    write_json(directory / 'lock.json', lock)
    seal(directory)
    return file_digest(directory / 'seal.json')


def base_directory(lock, override=None):
    return Path(lock['base']['directory'] if override is None else override).resolve()


def checked_lock(bundle, identity, *, check_current=True, base_bundle=None):
    bundle = Path(bundle).resolve()
    if file_digest(bundle / 'seal.json') != identity:
        raise ValueError('New B/D seal differs from trusted anchor')
    files = json.loads((bundle / 'seal.json').read_text('utf-8'))['files']
    observed = {p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file() and p.name != 'seal.json'}
    if observed != set(files):
        raise ValueError('B/D frozen membership changed')
    for name, expected in files.items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle) or (bundle / name).is_symlink() or file_digest(path) != expected['sha256']:
            raise ValueError('B/D frozen file changed')
    lock = json.loads((bundle / 'lock.json').read_text('utf-8'))
    version = lock['schema_version']
    if version not in ('industrial_bd_lock@1', 'industrial_bd_lock@2') or lock['official_test_used'] is not False:
        raise ValueError('Invalid B/D protocol')
    if check_current and version != 'industrial_bd_lock@2':
        raise ValueError('Archived B/D locks are read-only; executing requires a new portable source seal')
    if version == 'industrial_bd_lock@2':
        if file_digest(bundle / 'environment.json') != lock['environment_sha256']:
            raise ValueError('Frozen B/D environment differs from the lock')
        if check_current:
            verify_active_sources(lock['sources'])
            if platform_profile() != lock['platform_profile']:
                raise ValueError('Active FEDOT platform profile differs from the frozen manifest')
    base = base_directory(lock, base_bundle)
    if file_digest(base / 'lock.json') != lock['base']['lock_sha256']:
        raise ValueError('Archived TRAIN metadata changed')
    # Verify bytes without making the outer TRAIN partition available to search.
    if file_digest(base / 'seal.json') != lock['base']['seal']:
        raise ValueError('Archived TRAIN seal changed')
    base_files = json.loads((base / 'seal.json').read_text('utf-8'))['files']
    for name in lock['datasets']:
        member = f'inputs/{name}.npz'
        path = (base / member).resolve()
        if not path.is_relative_to(base / 'inputs') or file_digest(path) != base_files[member]['sha256']:
            raise ValueError(f'Archived TRAIN input changed: {name}')
    return lock
