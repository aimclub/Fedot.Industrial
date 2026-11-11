import argparse
from pathlib import Path
import json
import traceback


def main():
    parser = argparse.ArgumentParser(description='TRAIN-only mutation profile study; preview by default')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--generation-dir', type=Path)
    parser.add_argument('--generation-only', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--task', choices=('classification', 'regression', 'ts_forecasting'))
    parser.add_argument('--arm', choices=('baseline', 'structural'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.worker and not args.execute:
        parser.error('--worker requires --execute')
    if not args.execute and not args.worker:
        from .study import load_protocol
        print(json.dumps(load_protocol(), indent=2))
        return
    if args.worker:
        from .runner import run_trial
        try:
            run_trial(args.output_dir, args.task, args.arm, args.seed)
        except Exception as error:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / 'failure.json').write_text(json.dumps(dict(type=type(error).__name__,
                                                                          message=str(error), traceback=traceback.format_exc()), indent=2) + '\n', encoding='utf-8')
            raise
    elif args.generation_only:
        from .generation import run_generation
        run_generation(args.output_dir)
    else:
        if args.generation_dir is None:
            parser.error('--generation-dir is required for training')
        from .runner import run_matrix
        run_matrix(args.output_dir, args.generation_dir)


if __name__ == '__main__':
    main()
