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
    parser.add_argument('--seal', required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.freeze:
        if not args.execute or args.output_dir is None:
            parser.error('Freeze requires --execute and a new --output-dir')
        from .freezing import freeze
        print(json.dumps(dict(seal=freeze(args.bundle, args.seal, args.output_dir))))
        return
    from .freezing import checked_lock
    from .execution import trials, worker, supervise
    lock = checked_lock(args.bundle, args.seal)
    if not args.execute:
        print(json.dumps(dict(execution=False, jobs=len(trials(lock, args.admission)), admission=args.admission)))
        return
    if args.output_dir is None:
        parser.error('Execution requires --output-dir')
    if args.worker is None:
        supervise(args.output_dir, args.bundle, args.seal, args.admission, gate=args.gate)
    else:
        worker(args.output_dir, args.bundle, args.seal, args.worker, args.admission)


if __name__ == '__main__':
    from benchmark.industrial.experiments.mutation_quality.execution.runner import ProtocolViolation
    try:
        main()
    except (ProtocolViolation, ValueError) as error:
        import sys
        print(f'Protocol verification failed: {error}', file=sys.stderr)
        raise SystemExit(2)
