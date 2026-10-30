"""Preview by default; explicit preflight only reads TRAIN and writes a new report."""

import argparse
import json
from pathlib import Path

from .config import load_protocol


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    from .preflight import preflight, preview
    protocol = load_protocol()
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
