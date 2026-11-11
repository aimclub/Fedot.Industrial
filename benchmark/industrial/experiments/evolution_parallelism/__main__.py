"""Run one bounded real search. Preview is effect-free by default."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import traceback
import sys
import subprocess
from time import perf_counter

from .config import BACKENDS, TASKS, SearchConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS, default='classification')
    parser.add_argument('--backend', choices=BACKENDS, default='sequential')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--generations', type=int, default=15)
    parser.add_argument('--budget-seconds', type=float, default=1100)
    parser.add_argument('--lifecycle', choices=('batch', 'search'), default='batch')
    parser.add_argument('--protocol', choices=('generations', 'time'), default='generations')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--native-threads', type=int, default=1)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('examples/artifacts/cloud_bundle/evolution_parallelism/run'))
    args = parser.parse_args()
    config = SearchConfig(args.task, args.backend, args.generations, args.workers, args.budget_seconds,
                          lifecycle=args.lifecycle, protocol=args.protocol, pilot=args.pilot,
                          native_threads=args.native_threads)
    print(json.dumps(asdict(config), indent=2), flush=True)
    if not args.execute:
        return
    if args.output_dir.exists():
        parser.error('Output directory must not already exist; previous evidence is never overwritten')
    if not args.worker:
        from .supervisor import run_guarded
        start = perf_counter()
        try:
            code = run_guarded([sys.executable, '-m', __package__, *sys.argv[1:], '--worker'],
                               native_threads=config.native_threads, search_seconds=config.budget_seconds)
            status = 'finished'
        except subprocess.TimeoutExpired:
            code, status = 2, 'timed_out'
            args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.output_dir.is_dir():
            (args.output_dir / 'supervisor.json').write_text(json.dumps({
                'end_to_end_seconds': perf_counter() - start, 'hard_limit_seconds': 1200,
                'native_threads': config.native_threads, 'exit_code': code, 'status': status,
            }, indent=2) + '\n', encoding='utf-8')
        raise SystemExit(code)
    from .search import run_search, write_json
    try:
        result = run_search(config, args.output_dir)
    except Exception as error:
        if args.output_dir.is_dir():
            write_json(args.output_dir / 'failure.json', {'type': type(error).__name__, 'message': str(error),
                                                          'traceback': traceback.format_exc()})
        raise
    print(json.dumps(result, indent=2, default=str), flush=True)
    if result['status'] != 'succeeded':
        raise SystemExit(3)


if __name__ == '__main__':
    main()
