"""Seal complete plans and TRAIN tensors without evaluating a model."""

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import distributions
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from .config import load_protocol
from .preflight import file_digest, preflight


def write_json(path, value):
    with path.open('x', encoding='utf-8') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')


def array_record(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
        raise ValueError('Frozen model inputs must be finite numeric arrays without pickle')
    identity = sha256(str((value.shape, value.dtype.str)).encode() + value.tobytes()).hexdigest()
    return value, dict(shape=list(value.shape), dtype=value.dtype.str, sha256=identity)


def save_data(path, data):
    arrays, metadata = {}, {}
    containers = [(f'fold{index}_{role}', container) for index, pair in enumerate(data.folds)
                  for role, container in zip(('train', 'validation'), pair)]
    containers += [('refit', data.refit), ('control', data.test)]
    states = {}
    for role, container in containers:
        states[role] = dict(state=str(container.state), data_type=str(container.data_type),
                            task=str(container.task.task_type),
                            horizon=getattr(container, 'ts_forecast_horizon', None))
        for name in ('features', 'target', 'idx'):
            key = f'{role}_{name}'
            arrays[key], metadata[key] = array_record(getattr(container, name))
    with path.open('xb') as output:
        np.savez_compressed(output, **arrays)
    return dict(arrays=metadata, containers=states, source=data.source,
                outer_role='evaluation_only_inside_official_TRAIN', official_test_used=False)


def seal(directory):
    members = {path.relative_to(directory).as_posix(): dict(sha256=file_digest(path), bytes=path.stat().st_size)
               for path in sorted(directory.rglob('*')) if path.is_file() and path != directory / 'seal.json'}
    write_json(directory / 'seal.json', dict(schema_version='industrial_quality_seal@1', files=members))


def verify_freeze(directory, *, check_current=False, expected_seal_sha256=None):
    directory = Path(directory).resolve()
    if expected_seal_sha256 is not None and file_digest(directory / 'seal.json') != expected_seal_sha256:
        raise ValueError('Seal identity differs from the external trusted anchor')
    record = json.loads((directory / 'seal.json').read_text('utf-8'))
    if record['schema_version'] != 'industrial_quality_seal@1':
        raise ValueError('Unsupported seal')
    observed = {path.relative_to(directory).as_posix() for path in directory.rglob('*')
                if path.is_file() and path != directory / 'seal.json'}
    if observed != set(record['files']):
        raise ValueError('Frozen file membership changed')
    for name, expected in record['files'].items():
        original = directory / name
        path = original.resolve()
        if not path.is_relative_to(directory) or original.is_symlink():
            raise ValueError('Unsafe frozen member path')
        if path.stat().st_size != expected['bytes'] or file_digest(path) != expected['sha256']:
            raise ValueError(f'Frozen file changed: {name}')
    lock = json.loads((directory / 'lock.json').read_text('utf-8'))
    with ZipFile(directory / 'sources.zip') as archive:
        if set(archive.namelist()) != {item['member'] for item in lock['sources'].values()}:
            raise ValueError('Source archive membership changed')
        for path, item in lock['sources'].items():
            if sha256(archive.read(item['member'])).hexdigest() != item['sha256']:
                raise ValueError('Embedded source differs from lock')
            if check_current and file_digest(Path(path)) != item['sha256']:
                raise ValueError(f'Current source differs from frozen procedure: {path}')
    for name, expected in lock['datasets'].items():
        with np.load(directory / f'inputs/{name}.npz', allow_pickle=False) as saved:
            if set(saved.files) != set(expected['arrays']):
                raise ValueError('Frozen array membership changed')
            for key, metadata in expected['arrays'].items():
                if array_record(saved[key])[1] != metadata:
                    raise ValueError(f'Frozen array differs: {name}/{key}')
    return dict(verified=True, scenarios=len(lock['datasets']), jobs=len(lock['jobs']),
                sources=len(lock['sources']), current_sources_checked=check_current,
                seal_sha256=file_digest(directory / 'seal.json'), training_started=False)


def freeze_protocol(root, directory):
    import torch
    from .data import load_search_data
    from .runner import freeze_sources
    from .study import arm_policy
    from .sources import source_catalog

    root, directory = Path(root).resolve(), Path(directory).resolve()
    if Path.cwd().resolve() != root:
        raise ValueError('Freeze must run from the selected repository root')
    protocol = load_protocol()
    checked = preflight(root, protocol)
    if not checked['matrix_ready'] or any(row['sha256'] is None for row in checked['references'].values()):
        raise ValueError('All datasets, splits and reference configurations must be ready')
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'inputs').mkdir()
    flags = dict(threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                 deterministic=torch.are_deterministic_algorithms_enabled())
    frozen = freeze_sources()
    acceptance_path = root / 'benchmark/industrial/experiments/mutation_planning/population_study/protocol_20261019.json'
    frozen[str(acceptance_path)] = file_digest(acceptance_path)
    source_index = {}
    with ZipFile(directory / 'sources.zip', 'x', compression=ZIP_DEFLATED) as archive:
        for index, (path, expected) in enumerate(sorted(frozen.items())):
            content = Path(path).read_bytes()
            if sha256(content).hexdigest() != expected:
                raise RuntimeError('Source changed before archival')
            member = f'sources/{index:04d}{Path(path).suffix}'
            archive.writestr(member, content)
            source_index[path] = dict(sha256=expected, member=member)
    datasets = {}
    for scenario, reviewed in zip(protocol.scenarios, checked['scenarios']):
        data = load_search_data(scenario, protocol, root)
        if any(data.source[key] != reviewed[key] for key in reviewed):
            raise RuntimeError('TRAIN source or split changed after preflight')
        datasets[scenario.name] = save_data(directory / f'inputs/{scenario.name}.npz', data)
    training = json.loads((root / protocol.training_reference).read_text('utf-8'))
    technical_path = root / 'examples/artifacts/cloud_bundle/mutation_quality_technical_20261030/pilot_01/jobs.json'
    technical = json.loads(technical_path.read_text('utf-8'))
    expected_flags = technical[0]['verification']['torch_flags']
    if len(technical) != 9 or any(row['exit_code'] != 0 or row['verification']['torch_flags'] != expected_flags
                                  for row in technical):
        raise ValueError('Technical prerequisite is incomplete or inconsistent')
    policies = {arm.value: arm_policy(arm).to_record() for arm in protocol.arms}
    tasks = {row.name: row.task for row in protocol.scenarios}
    lock = dict(schema_version='industrial_mutation_quality_lock@1',
                created_at=datetime.now(timezone.utc).isoformat(),
                git_revision_context=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                normative_identity='embedded source bytes and sealed TRAIN arrays, not HEAD alone',
                full_series_enabled=False, training_started=False, official_test_used=False,
                protocol=asdict(protocol), policies=policies, training_reference=training,
                scenario_settings={row.name: {**training['tasks'][row.task], 'dataset': row.name}
                                   for row in protocol.scenarios},
                acceptance_reference=json.loads(acceptance_path.read_text('utf-8'))['acceptance'],
                source_catalog=source_catalog(), sources=source_index, datasets=datasets,
                jobs=[dict(scenario=name, task=tasks[name], arm=arm, seed=seed,
                           search_seconds=protocol.budget_seconds, candidate_seconds=protocol.candidate_seconds,
                           minimum_generations=protocol.minimum_generations, stop_protocol='time',
                           full_process_seconds=training['full_process_seconds'], data=f'inputs/{name}.npz')
                      for name, arm, seed in protocol.jobs()],
                scoring=dict(classification='accuracy', regression='rmse', ts_forecasting='rmse',
                             direction=dict(classification='maximize', regression='minimize',
                                            ts_forecasting='minimize'),
                             comparisons=['expanded_fixed-baseline', 'expanded_staged-expanded_fixed'],
                             outer_control='evaluation only; never tune against it',
                             failures='retain all failures and incomplete generations; no success-only filtering',
                             grouping={row.name: row.group for row in protocol.scenarios}),
                execution=dict(backend='dask', lifecycle='search', workers=protocol.workers,
                               native_threads=protocol.native_threads, concurrent_searches=1,
                               torch_flags=expected_flags, full_runner_status='disabled; technical runner only'),
                technical_reference=dict(path=str(technical_path), sha256=file_digest(technical_path)),
                freeze_process_torch_flags=flags)
    write_json(directory / 'lock.json', lock)
    write_json(directory / 'preflight.json', checked)
    write_json(directory / 'environment.json', dict(python=sys.version, executable=sys.executable,
               packages={dist.metadata['Name']: dist.version for dist in distributions()}, torch_flags=flags))
    if frozen != {path: file_digest(Path(path)) for path in frozen}:
        raise RuntimeError('Sources changed while freezing')
    if flags != dict(threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                     deterministic=torch.are_deterministic_algorithms_enabled()):
        raise RuntimeError('Torch settings changed while freezing')
    seal(directory)
    return verify_freeze(directory, check_current=True)
