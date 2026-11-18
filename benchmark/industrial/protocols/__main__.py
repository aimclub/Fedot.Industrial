"""Resolve and check a protocol manifest without executing it."""

import argparse
import json
from pathlib import Path

from . import MissingRoastError, ProtocolError, preflight_protocol, resolve_protocol


def _unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ProtocolError('JSON manifest contains duplicate object keys')
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--data-root', type=Path)
    args = parser.parse_args(argv)
    if args.data_root is not None and not args.preflight:
        parser.error('--data-root requires --preflight')

    try:
        with args.manifest.open(encoding='utf-8') as handle:
            payload = json.load(handle, object_pairs_hook=_unique_object)
    except OSError as error:
        print(json.dumps({'status': 'unavailable', 'detail': str(error)}))
        return 2
    except ValueError as error:
        print(json.dumps({'status': 'incompatible', 'detail': str(error)}))
        return 2

    try:
        plan = resolve_protocol(payload)
        if not args.preflight:
            print(json.dumps(plan.to_dict(), ensure_ascii=True, indent=2, sort_keys=True))
            return 0
        report = preflight_protocol(plan, args.data_root)
    except MissingRoastError as error:
        print(json.dumps({'status': 'unavailable', 'detail': str(error)}))
        return 2
    except (ProtocolError, OSError) as error:
        print(json.dumps({'status': 'incompatible', 'detail': str(error)}))
        return 2
    print(json.dumps({'plan_id': plan.plan_id, **report.to_dict()}, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report.ready else 1


if __name__ == '__main__':
    raise SystemExit(main())
