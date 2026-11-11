"""Preview by default; freeze, admission and full series are distinct commands."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--admission', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--base-bundle', type=Path, help='Local path to the unchanged archived TRAIN bundle')
    parser.add_argument('--seal', required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--admission-first', action='store_true', help='Run a new admission before the full matrix')
    parser.add_argument('--resume-after-cleanup', type=int, metavar='TRIAL_INDEX',
                        help='Verify the blocked trial cleanup and run only its remaining suffix')
    parser.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.admission_first and (not args.run or args.worker is not None or args.gate is None):
        parser.error('--admission-first requires --run, --gate and supervision, not --worker')
    if args.resume_after_cleanup is not None and (not args.run or not args.execute or args.worker is not None
                                                  or args.admission_first):
        parser.error('--resume-after-cleanup requires an existing full-series run with --run --execute')
    if args.freeze:
        if not args.execute or args.output_dir is None or args.base_bundle is not None:
            parser.error('Freeze requires --execute, the archived TRAIN bundle and a new --output-dir')
        from .freezing import freeze
        print(json.dumps(dict(seal=freeze(args.bundle, args.seal, args.output_dir))))
        return
    from .freezing import checked_lock
    from .execution import trials, worker, supervise
    lock = checked_lock(args.bundle, args.seal, check_current=args.execute, base_bundle=args.base_bundle)
    if not args.execute:
        print(json.dumps(dict(execution=False, lock_schema=lock['schema_version'],
                              jobs=len(trials(lock, args.admission)), admission=args.admission,
                              admission_first=args.admission_first)))
        return
    if args.output_dir is None:
        parser.error('Execution requires --output-dir')
    if args.worker is None:
        if args.admission_first:
            supervise(args.gate, args.bundle, args.seal, True, base_bundle=args.base_bundle)
        supervise(args.output_dir, args.bundle, args.seal, args.admission, gate=args.gate,
                  resume_after_cleanup=args.resume_after_cleanup, base_bundle=args.base_bundle)
    else:
        worker(args.output_dir, args.bundle, args.seal, args.worker, args.admission, base_bundle=args.base_bundle)


if __name__ == '__main__':
    from benchmark.industrial.experiments.mutation_quality.execution.runner import ProtocolViolation
    try:
        main()
    except (ProtocolViolation, ValueError, OSError) as error:
        import sys
        print(f'Protocol verification failed: {error}', file=sys.stderr)
        raise SystemExit(2)
