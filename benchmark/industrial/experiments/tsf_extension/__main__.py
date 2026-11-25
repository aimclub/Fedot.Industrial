"""Report the TSF extension matrix without opening a dataset or running training."""

import argparse
import json
from pathlib import Path

from .planning import VERSION, load_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    actions.add_parser('plan').add_argument('--protocol', choices=(VERSION,), required=True)
    seal_command = actions.add_parser('seal')
    seal_command.add_argument('--data-root', type=Path, required=True)
    seal_command.add_argument('--previous-bundle', type=Path, required=True)
    seal_command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'seal':
        from .inputs import seal
        result = seal(args.data_root, args.previous_bundle, args.output)
        print(json.dumps({'searches': result['searches'], 'series': [
            [row['collection']['id'], row['id']] for row in result['rows']
        ]}, indent=2))
        return
    collections, seeds = load_profile()
    print(json.dumps({
        'version': VERSION,
        'status': 'requires_verified_inputs_and_executor',
        'official_test_used': False,
        'collections': [{'id': row.id, 'horizon': row.horizon, 'minimum_train': row.min_history,
                         'required_train_file': row.filename, 'planned_series': 4} for row in collections],
        'seeds': seeds,
        'planned_searches': len(collections) * 4 * len(seeds),
    }, indent=2))


if __name__ == '__main__':
    main()
