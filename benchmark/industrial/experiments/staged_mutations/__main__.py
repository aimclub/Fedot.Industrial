import argparse
import json
from pathlib import Path

from .config import build_config, load_protocol


def main():
    parser = argparse.ArgumentParser(description='Staged mutation protocol: preview or no-training audit')
    parser.add_argument('--mode', choices=('fixed', 'staged'), default='staged')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--task', choices=('classification', 'regression', 'ts_forecasting'), default='classification')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.train:
        from .training import load_training_protocol
        from .training_runner import run_matrix, run_trial
        if args.audit:
            parser.error('--audit and --train cannot be combined')
        if not args.execute:
            print(json.dumps({'protocol': load_training_protocol(), 'execution': False}, indent=2))
            return
        if args.output_dir is None:
            parser.error('--execute requires --output-dir')
        if args.worker:
            run_trial(args.output_dir, args.task, args.mode, args.seed, pilot=args.pilot)
        else:
            run_matrix(args.output_dir, pilot=args.pilot)
    elif args.audit:
        if args.output_dir is None:
            parser.error('--audit requires a new --output-dir')
        from .audit import audit
        result = audit(args.output_dir)
        print(json.dumps([dict(task=r['task'], admissible=r['admissible_after_runtime'], families=r['families'])
                          for r in result['rows']], indent=2))
    else:
        print(json.dumps(dict(protocol=load_protocol(), config=build_config(args.mode).to_record()), indent=2))


if __name__ == '__main__':
    main()
