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
from .study import policy_for


def freeze(base, base_seal, directory):
    import_module('fedot')  # Distribution visibility must match workers after FEDOT initialization.
    import torch
    base, directory = Path(base).resolve(), Path(directory).resolve()
    verify_freeze(base, check_current=False, expected_seal_sha256=base_seal)
    original = json.loads((base / 'lock.json').read_text('utf-8'))
    lock = deepcopy(original)
    lock['schema_version'] = 'industrial_bd_lock@1'
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
    lock['admission_scenarios'] = ['ECG200', 'Tecator', 'Airline']
    lock['scoring']['classification'] = 'balanced_accuracy'
    lock['scoring']['comparisons'] = ['cyclic-fixed']
    paths = {Path(p) for p in original['sources']}
    paths.update(p.resolve() for root in ('fedot_ind', 'benchmark/industrial/experiments')
                 for pattern in ('*.py', '*.json') for p in Path(root).rglob(pattern))
    directory.mkdir(parents=True, exist_ok=False)
    sources = {}
    with ZipFile(directory / 'sources.zip', 'x', compression=ZIP_DEFLATED) as archive:
        for i, path in enumerate(sorted(paths)):
            data = path.read_bytes()
            member = f'sources/{i:04d}{path.suffix}'
            archive.writestr(member, data)
            sources[str(path.resolve())] = dict(sha256=sha256(data).hexdigest(), member=member)
    lock['sources'] = sources
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
    write_json(directory / 'lock.json', lock)
    write_json(directory / 'data_preflight.json', dict(scenarios=checked, fit_calls=0, test_used=False))
    write_json(directory / 'environment.json', dict(python=sys.version, executable=sys.executable,
               packages={dist.metadata['Name']: dist.version for dist in distributions()}, torch_flags=flags))
    seal(directory)
    return file_digest(directory / 'seal.json')


def checked_lock(bundle, identity):
    bundle = Path(bundle).resolve()
    if file_digest(bundle / 'seal.json') != identity:
        raise ValueError('New B/D seal differs from trusted anchor')
    files = json.loads((bundle / 'seal.json').read_text('utf-8'))['files']
    observed = {p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file() and p.name != 'seal.json'}
    if observed != set(files):
        raise ValueError('B/D frozen membership changed')
    for name, expected in files.items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle) or file_digest(path) != expected['sha256']:
            raise ValueError('B/D frozen file changed')
    lock = json.loads((bundle / 'lock.json').read_text('utf-8'))
    if lock['schema_version'] != 'industrial_bd_lock@1' or lock['official_test_used'] is not False:
        raise ValueError('Invalid B/D protocol')
    for path, expected in lock['sources'].items():
        if file_digest(Path(path)) != expected['sha256']:
            raise ValueError(f'Current source differs from new B/D seal: {path}')
    base = Path(lock['base']['directory'])
    if file_digest(base / 'lock.json') != lock['base']['lock_sha256']:
        raise ValueError('Archived TRAIN metadata changed')
    # Verify bytes without making the outer TRAIN partition available to search.
    base_files = json.loads((base / 'seal.json').read_text('utf-8'))['files']
    if file_digest(base / 'seal.json') != lock['base']['seal']:
        raise ValueError('Archived TRAIN seal changed')
    for name in lock['datasets']:
        member = f'inputs/{name}.npz'
        if file_digest(base / member) != base_files[member]['sha256']:
            raise ValueError(f'Archived TRAIN input changed: {name}')
    return lock
