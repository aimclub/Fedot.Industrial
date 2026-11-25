"""Resolve, check or report on a protocol without training a model."""

import argparse
import json
from pathlib import Path

from . import MissingRoastError, preflight_protocol, resolve_protocol
from .json_io import read_protocol_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--report-evidence', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if args.data_root is not None and not args.preflight:
        parser.error('--data-root requires --preflight')
    if (args.output_dir is None) != (args.report_evidence is None):
        parser.error('--report-evidence and --output-dir must be supplied together')

    try:
        payload = read_protocol_json(args.manifest)
    except OSError as error:
        print(json.dumps({'status': 'unavailable', 'detail': str(error)}))
        return 2
    except ValueError as error:
        print(json.dumps({'status': 'incompatible', 'detail': str(error)}))
        return 2

    try:
        plan = resolve_protocol(payload)
        if args.report_evidence is not None:
            from ..evaluation.protocol_report import report_from_evidence, write_protocol_report

            report = report_from_evidence(plan, read_protocol_json(args.report_evidence))
            write_protocol_report(report, args.output_dir, inputs=(args.manifest, args.report_evidence))
            print(json.dumps({'plan_id': plan.plan_id, 'success': report.successful, 'expected': report.expected}))
            return 0
        if not args.preflight:
            print(json.dumps(plan.to_dict(), ensure_ascii=True, indent=2, sort_keys=True))
            return 0
        report = preflight_protocol(plan, args.data_root)
    except MissingRoastError as error:
        print(json.dumps({'status': 'unavailable', 'detail': str(error)}))
        return 2
    except (ValueError, OSError) as error:
        print(json.dumps({'status': 'incompatible', 'detail': str(error)}))
        return 2
    print(json.dumps({'plan_id': plan.plan_id, **report.to_dict()}, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report.ready else 1


if __name__ == '__main__':
    raise SystemExit(main())
