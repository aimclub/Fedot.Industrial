"""Preview by default. Full execution requires an intact smoke gate and explicit intent."""

import argparse
import json
from pathlib import Path
import sys

from .plans import trial_plan
from .runner import ProtocolViolation, checked_lock, supervise, worker


def main():
    parser = argparse.ArgumentParser(__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--seal', required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--recover-stale-lock', action='store_true')
    parser.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.start_index < 0 or (args.smoke and args.start_index):
        parser.error('--start-index is only supported for a full-series suffix')
    lock = checked_lock(args.bundle, args.seal)
    if not args.execute:
        print(json.dumps(dict(execution=False, jobs=len(trial_plan(lock, smoke=args.smoke)[args.start_index:]),
                              smoke=args.smoke, start_index=args.start_index)))
        return
    if args.output_dir is None or (args.run and args.worker is None and args.gate is None):
        parser.error('Execution requires --output-dir and full supervision requires --gate')
    if args.worker is not None:
        worker(args.output_dir, args.bundle, args.seal, args.worker, smoke=args.smoke)
    else:
        supervise(args.output_dir, args.bundle, args.seal, smoke=args.smoke, gate=args.gate,
                  start_index=args.start_index, recover_stale_lock=args.recover_stale_lock)


if __name__ == '__main__':
    try:
        main()
    except ProtocolViolation as error:
        print(f'PROTOCOL VIOLATION: {error}', file=sys.stderr, flush=True)
        sys.exit(2)
