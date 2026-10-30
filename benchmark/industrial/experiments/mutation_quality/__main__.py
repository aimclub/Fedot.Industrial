"""Preview by default; explicit preflight only reads TRAIN and writes a new report."""

import argparse
import json
from pathlib import Path

from .config import load_protocol


def main():
    parser = argparse.ArgumentParser(__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--technical', action='store_true')
    mode.add_argument('--prepare-data', action='store_true')
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--verify-freeze', action='store_true')
    parser.add_argument('--expected-seal')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--scenario')
    parser.add_argument('--arm', choices=('baseline', 'expanded_fixed', 'expanded_staged'))
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    from .preflight import preflight, preview
    protocol = load_protocol()
    if args.prepare_data or args.freeze or args.verify_freeze:
        if args.expected_seal and not args.verify_freeze:
            parser.error('--expected-seal requires --verify-freeze')
        if args.worker or args.scenario or args.arm or args.execute:
            parser.error('Data preparation and freezing cannot be combined with training flags')
        if args.output_dir is None:
            parser.error('Data preparation and freezing require --output-dir')
        if args.prepare_data:
            from .sources import prepare_sources
            result = prepare_sources(args.output_dir)
        elif args.freeze:
            from .freeze import freeze_protocol
            result = freeze_protocol(Path(__file__).resolve().parents[4], args.output_dir)
        else:
            from .freeze import verify_freeze
            result = verify_freeze(args.output_dir, check_current=True, expected_seal_sha256=args.expected_seal)
        print(json.dumps(result, indent=2))
        return
    if args.expected_seal:
        parser.error('--expected-seal requires --verify-freeze')
    if args.technical:
        from .runner import run_technical, run_trial, technical_jobs, technical_settings
        if not args.execute:
            print(json.dumps(dict(settings=technical_settings(), jobs=technical_jobs(), training_started=False), indent=2))
            return
        if args.output_dir is None:
            parser.error('--technical --execute requires --output-dir')
        if args.worker:
            run_trial(args.output_dir, args.scenario, args.arm, technical_settings()['seed'])
        else:
            if args.scenario is not None or args.arm is not None:
                parser.error('Technical supervisor runs only the fixed nine-trial matrix')
            run_technical(args.output_dir)
        return
    if args.execute or args.worker or args.scenario or args.arm:
        parser.error('Training flags require --technical')
    if not args.preflight:
        print(json.dumps(preview(protocol), indent=2))
        return
    if args.output_dir is None:
        parser.error('--preflight requires --output-dir')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = preflight(Path(__file__).resolve().parents[4], protocol)
    (args.output_dir / 'preflight.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'matrix_ready': result['matrix_ready'],
                      'scenarios': [(row['name'], row['status']) for row in result['scenarios']]}, indent=2))


if __name__ == '__main__':
    main()
