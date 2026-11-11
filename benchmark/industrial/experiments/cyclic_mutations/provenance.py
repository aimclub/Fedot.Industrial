"""Identify the sources selected by this process, independently of checkout paths."""
from importlib.util import find_spec
import json
from pathlib import Path
import sys

from benchmark.industrial.experiments.mutation_quality.execution.runner import source_identity
from benchmark.industrial.experiments.mutation_quality.preflight import file_digest


_SOURCE_SUFFIXES = {'.py', '.pyx', '.json', '.yaml', '.yml', '.toml', '.so', '.pyd'}
_PLATFORM_POLICY = 'tools/platform_support/compatibility.json'


def _package_root(name):
    spec = find_spec(name)
    if spec is None or spec.origin is None or spec.submodule_search_locations is None:
        raise ValueError(f'{name} must be imported from a filesystem package')
    root = Path(spec.origin).resolve().parent
    if {Path(location).resolve() for location in spec.submodule_search_locations} != {root}:
        raise ValueError(f'Ambiguous source package: {name}')
    return root


def _check_loaded_modules(packages):
    for name, module in tuple(sys.modules.items()):
        root = packages.get(name.split('.', 1)[0])
        if root is None or module is None:
            continue
        filename = getattr(module, '__file__', None)
        locations = (filename,) if filename else getattr(module, '__path__', ())
        if not locations or any(not Path(path).resolve().is_relative_to(root) for path in locations):
            raise ValueError(f'Loaded module {name} is outside the selected source package')


def _execution_roots():
    checkout = Path(__file__).resolve().parents[4]
    if (Path.cwd().resolve() != checkout or _package_root('benchmark') != checkout / 'benchmark'
            or _package_root('fedot_ind') != checkout / 'fedot_ind'):
        raise ValueError('Execute B/D from the checkout supplying benchmark and fedot_ind')
    fedot, golem = _package_root('fedot'), _package_root('golem')
    _check_loaded_modules(dict(benchmark=checkout / 'benchmark', fedot_ind=checkout / 'fedot_ind',
                               fedot=fedot, golem=golem))
    return {'industrial': checkout, 'fedot': fedot, 'golem': golem}


def source_inventory():
    roots = _execution_roots()
    sources = {}
    for group, root in roots.items():
        paths = (root / 'fedot_ind', root / 'benchmark' / 'industrial') if group == 'industrial' else (root,)
        members = [p for path in paths for p in path.rglob('*') if p.suffix in _SOURCE_SUFFIXES and p.is_file()]
        if group == 'industrial':
            members.extend(root / name for name in ('benchmark/__init__.py', _PLATFORM_POLICY, 'pyproject.toml'))
        for member in members:
            if not member.is_file() or member.is_symlink() or not member.resolve().is_relative_to(root):
                raise ValueError(f'Unsafe or missing active source: {member}')
            sources[f'{group}/{member.relative_to(root).as_posix()}'] = member
    return dict(sorted(sources.items()))


def verify_active_sources(expected):
    active = source_inventory()
    if set(expected) != set(active):
        raise ValueError('Active B/D source membership differs from the frozen manifest')
    for name, path in active.items():
        if file_digest(path) != expected[name]:
            raise ValueError(f'Active B/D source differs from the frozen manifest: {name}')


def platform_profile():
    checkout = Path(__file__).resolve().parents[4]
    policy = json.loads((checkout / _PLATFORM_POLICY).read_text('utf-8'))
    name = policy['default_profile']
    # The configured reference is descriptive; the active FEDOT files are hashed separately.
    return dict(name=name, configured_fedot_sha=policy['profiles'][name]['sha'])


def runtime_record(lock, identity):
    return dict(schema_version='industrial_bd_runtime@1', seal=identity,
                source_manifest_sha256=source_identity(lock['sources']), environment_sha256=lock['environment_sha256'],
                platform_profile=lock['platform_profile'], frozen_revision_context=lock['git_revision_context'])
