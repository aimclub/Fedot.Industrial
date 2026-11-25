"""Plan, prepare, run and assess the separately authorized parts of the TSC pilot."""

import argparse
import json
from pathlib import Path
from subprocess import TimeoutExpired
import sys

from .protocol import TSC_VERSION, TSER_VERSION, tsc_protocol, tser_preflight_plan


def _layout(bundle, output, mode, protocol):
    from .data import canonical_bytes, digest
    from .execution import read_record, write_record

    seal = read_record(bundle / 'seal.json')
    expected = protocol.as_payload()
    if seal.get('protocol') == TSER_VERSION:
        from .regression_inputs import tser_profile
        expected = tser_profile()
    if (seal.get('official_test_used') is not False or seal.get('protocol') != expected['version']
            or seal.get('protocol_sha256') != digest(canonical_bytes(expected))):
        raise ValueError('A matching TRAIN-only seal is required')
    if mode == 'run':
        output.mkdir(parents=True, exist_ok=False)
        write_record(output / 'run.json', dict(seal_sha256=digest(canonical_bytes(seal)), device='cuda:0'))
    elif read_record(output / 'run.json')['seal_sha256'] != digest(canonical_bytes(seal)):
        raise ValueError('The search output belongs to a different seal')
    return seal


def _execute(mode, bundle, output, protocol, admission=None):
    from ..evolution_parallelism.supervisor import run_guarded
    from .data import canonical_bytes, digest
    from .execution import read_record, write_record

    if mode == 'run':
        seal = read_record(bundle / 'seal.json')
        if read_record(admission / 'run.json')['seal_sha256'] != digest(canonical_bytes(seal)):
            raise ValueError('CUDA admission belongs to a different seal')
        rows = read_record(admission / seal['datasets'][0]['dataset'] / 'admit' / 'cuda_admission.json')['results']
        if len(rows) != 3 or not all(
            row['status'] == 'succeeded' and row['value']['folds'][0]['training']
            and all(item['device'] == 'cuda:0' for item in row['value']['folds'][0]['training'])
            for row in rows
        ):
            raise ValueError('All three neural operations must pass CUDA admission')
    seal = _layout(bundle, output, mode, protocol)
    budget = protocol.search
    rows = []
    for source in seal['datasets']:
        name = source['dataset']
        stages = ('calibrate', *map(str, budget.seeds)) if mode == 'run' else tuple(map(str, budget.seeds))
        for stage in stages:
            action = 'assess' if mode == 'assess' else 'run'
            command = (sys.executable, '-m', __package__, '_stage', action, str(bundle.resolve()),
                       str(output.resolve()), name, stage)
            try:
                code = run_guarded(command, timeout_seconds=budget.process_seconds,
                                   search_seconds=budget.search_seconds)
            except TimeoutExpired:
                code = 124
            rows.append(dict(dataset=name, stage=stage, exit_code=code))
            write_record(output / f'{mode}_status.json', rows)
            if code and stage == 'calibrate':
                break
    return 1 if any(row['exit_code'] for row in rows) else 0


def _stage(args, protocol):
    from .assessment import assess
    from .execution import calibrate, cuda_admission, read_record, search

    from fedot_ind.core.optimizer.evaluation.dask_config import DaskExecutionConfig
    from fedot_ind.core.optimizer.evaluation.dask_executor import DaskEvaluationSession
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('The TSC pilot requires a CUDA device. No fallback to CPU is allowed.')
    seal = read_record(args.bundle / 'seal.json')
    regression = seal['protocol'] == TSER_VERSION
    record, = (value for value in seal['datasets'] if value['dataset'] == args.dataset)
    data = args.bundle / args.dataset
    root = args.output / args.dataset
    root.mkdir(exist_ok=True)
    run_budget = protocol.search
    path = root / args.stage
    if args.mode == 'assess':
        return assess(data, path, record, read_record(root / 'calibrate' / 'recipes.json'), run_budget)
    path.mkdir()
    # All isolated workers can reach this GPU; admission is bounded by the process count.
    with DaskEvaluationSession(2, DaskExecutionConfig()) as session:
        if args.stage == 'admit':
            return 0 if cuda_admission(session, data / 'development', path, run_budget, protocol.profile.split_seed,
                                       record['development'], regression=regression) else 1
        if args.stage == 'calibrate':
            calibrate(session, data / 'development', path, run_budget, protocol.profile.split_seed,
                      record['development'], regression=regression)
        else:
            if int(args.stage) not in run_budget.seeds:
                raise ValueError('Unexpected search seed')
            row = search(session, data / 'development', path, run_budget, int(args.stage),
                         record['development'], read_record(root / 'calibrate' / 'recipes.json'), regression=regression)
            return 0 if row['status'] == 'complete' else 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('plan').add_argument('--protocol', choices=(TSC_VERSION, TSER_VERSION), required=True)
    seal = commands.add_parser('seal')
    seal.add_argument('--data-root', type=Path, required=True)
    seal.add_argument('--output', type=Path, required=True)
    seal.add_argument('--tser-evidence', type=Path)
    for label in ('run', 'assess'):
        command = commands.add_parser(label)
        command.add_argument('--bundle', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        if label == 'run':
            command.add_argument('--admission', type=Path, required=True)
    admission = commands.add_parser('admit')
    admission.add_argument('--bundle', type=Path, required=True)
    admission.add_argument('--output', type=Path, required=True)
    stage = commands.add_parser('_stage')
    stage.add_argument('mode', choices=('run', 'assess'))
    stage.add_argument('bundle', type=Path)
    stage.add_argument('output', type=Path)
    stage.add_argument('dataset')
    stage.add_argument('stage')
    args = parser.parse_args(argv)
    if args.action == 'plan' and args.protocol == TSER_VERSION:
        print(json.dumps(tser_preflight_plan(), indent=2))
        return 0
    protocol = tsc_protocol()
    if args.action == 'plan':
        print(json.dumps(dict(protocol=protocol.version, tasks=protocol.tasks(), official_test_used=False), indent=2))
        return 0
    if args.action == 'seal':
        from .data import seal_tsc
        if args.tser_evidence is None:
            receipt = seal_tsc(protocol, args.data_root, args.output)
        else:
            from .regression_inputs import seal_regression
            receipt = seal_regression(args.data_root, args.output, args.tser_evidence)
        print(json.dumps(receipt, indent=2))
        return 0
    if args.action == '_stage':
        return _stage(args, protocol)
    if args.action == 'admit':
        from ..evolution_parallelism.supervisor import run_guarded
        seal = _layout(args.bundle, args.output, 'run', protocol)
        command = (sys.executable, '-m', __package__, '_stage', 'run', str(args.bundle.resolve()),
                   str(args.output.resolve()), seal['datasets'][0]['dataset'], 'admit')
        return run_guarded(command, timeout_seconds=protocol.search.process_seconds,
                           search_seconds=protocol.search.search_seconds)
    return _execute(args.action, args.bundle, args.output, protocol, getattr(args, 'admission', None))


if __name__ == '__main__':
    sys.exit(main())
