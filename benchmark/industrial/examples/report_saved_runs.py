"""Replay the DATA-10 public receipts without loading datasets or models."""

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
from html import escape
from io import StringIO
from itertools import product
import math
from pathlib import Path

from benchmark.industrial.evaluation.protocol_report import render_html_table, write_csv_table
from benchmark.industrial.protocols.json_io import parse_protocol_json, read_protocol_json
from benchmark.industrial.protocols.schema import ProtocolError, mapping, string


_ROOT = Path(__file__).with_name('saved_protocol_report')
_HEADERS = ('cohort', 'task', 'metric', 'dataset', 'model', 'unit', 'status', 'value',
            'source', 'artifact', 'reason')


@dataclass(frozen=True)
class Cell:
    dataset: str
    model: str
    unit: str


@dataclass(frozen=True)
class Outcome:
    cell: Cell
    status: str
    value: float | None
    artifact: str
    reason: str = ''

    def __post_init__(self):
        if self.value is not None and not math.isfinite(self.value):
            raise ProtocolError(f'Non-finite score for {self.cell}')
        if self.status == 'success' and self.value is None:
            raise ProtocolError(f'Missing score for successful outcome {self.cell}')


def _read_receipt(root, source, *, csv_file=False):
    path = root / source['file']
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return () if csv_file else None
    if sha256(data).hexdigest() != source['sha256']:
        raise ProtocolError(f'Saved receipt checksum differs: {path.name}')
    if csv_file:
        return tuple(csv.DictReader(StringIO(data.decode('utf-8-sig'), newline='')))
    return mapping(parse_protocol_json(data.decode('utf-8')), f'Saved receipt {path.name}')


def _stop_reasons(rows):
    indexed = {}
    for row in rows:
        attempt = string(row.get('index'), 'B/D attempt index')
        reason = string(row.get('cause'), 'B/D stop reason')
        if indexed.setdefault(attempt, reason) != reason:
            raise ProtocolError(f'Conflicting stop reasons for B/D attempt {attempt}')
    return indexed


def _index(items):
    indexed = {}
    for row in items:
        if indexed.setdefault(row.cell, row) != row:
            raise ProtocolError(f'Conflicting outcomes: {row.cell}')
    return indexed


def _with_coverage(expected, items):
    rows = _index(items)
    if set(rows) - set(expected):
        raise ProtocolError('Saved outcomes outside the declared cohort')
    return tuple(rows.get(cell, Outcome(cell, 'unavailable', None, '', 'no_saved_outcome'))
                 for cell in expected)


def _search_outcomes(spec, attempts, stops):
    for row in attempts:
        if row['task'] != spec['source_task']:
            continue
        status = {'succeeded': 'success', 'incomplete': 'incomplete', 'failed': 'failed'}[row['status']]
        if row['passed'] != str(status == 'success'):
            raise ProtocolError(f"Inconsistent search status for {row['path']}")
        reason = '' if status == 'success' else stops.get(row['index'], 'stop_reason_not_saved')
        yield Outcome(Cell(row['dataset'], row['arm'], row['seed']), status,
                      float(row['control_score']) if row['control_score'] else None, row['path'], reason)


def _time_outcomes(spec, receipt):
    if receipt is None:
        return
    release = string(spec.get('dataset_release'), 'TIME dataset release')
    if receipt.get('dataset_release') != release:
        raise ProtocolError('TIME receipt does not match the declared dataset release')
    for window in receipt.get('windows', ()):
        if (window['status'] != 'success' or window['mode'] != 'zero_shot'
                or window['horizon'] != spec['horizon'] or window['model_release'] != spec['model_release']):
            raise ProtocolError('TIME window does not match the declared frozen forecast')
        for model, metric in spec['fields'].items():
            yield Outcome(Cell(window['datasets'][0], model, window['origin']), 'success', window[metric],
                          f"windows/{window['origin']}")


def _nab_outcomes(spec, receipt):
    if receipt is not None:
        result = receipt['nab']
        if result['revision'] != spec['dataset_release']:
            raise ProtocolError('NAB receipt does not match the declared release')
        yield Outcome(Cell(result['stream'], spec['models'][0], spec['units'][0]), 'success',
                      result[spec['metric']], 'nab')


def _report_rows(spec, items):
    expected = tuple(Cell(*parts) for parts in product(spec['datasets'], spec['models'], spec['units']))
    if len(expected) != len(set(expected)):
        raise ProtocolError(f"Duplicate cells in {spec['id']}")
    return _with_coverage(expected, items)


def _pair_coverage(spec, rows):
    if len(spec['models']) != 2:
        return ''
    first, second = spec['models']
    successful = {row.cell for row in rows if row.status == 'success'}
    pairs = [
        (dataset, len(spec['units']), sum(Cell(dataset, first, unit) in successful
                                          and Cell(dataset, second, unit) in successful for unit in spec['units']))
        for dataset in spec['datasets']
    ]
    return render_html_table(('Dataset', 'Expected pairs', 'Successful pairs'), pairs)


def _section(spec, rows, table, source):
    details = (
        ('Evaluation', spec['description']),
        ('Source revision', source['revision']),
        ('Saved snapshot', source['recorded_at']),
        ('Source SHA-256', source['sha256']),
        ('Scoring unit', spec['unit_description']),
    )
    if 'data_seal' in source:
        details += (('Frozen data and protocol seal', source['data_seal']),)
    context = ''.join(f'<dt>{escape(key)}</dt><dd>{escape(value)}</dd>' for key, value in details)
    status = ', '.join(f'{key}: {sum(row.status == key for row in rows)}'
                       for key in ('success', 'incomplete', 'failed', 'unavailable'))
    coverage = _pair_coverage(spec, rows)
    return (f'<section id="{escape(spec["id"])}"><h2>{escape(spec["title"])}</h2>'
            f'<p>{len(rows)} expected; {sum(row.value is not None for row in rows)} scored. {status}.</p>'
            f'<dl>{context}</dl>{coverage}'
            f'<div class="table">{render_html_table(_HEADERS, table)}</div></section>')


_PAGE = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Industrial saved benchmark results</title>
<style>
body { font: 15px/1.5 system-ui, sans-serif; color: #20282a; background: #fafbfa;
       margin: 2rem auto; padding: 0 1rem; max-width: 90rem; }
h1 { font-size: 1.9rem; } h2 { font-size: 1.35rem; }
a { color: #0a6468; } nav a { display: inline-block; margin: 0 .8rem .4rem 0; }
section { margin: 2rem 0; padding-top: .6rem; border-top: 1px solid #d8dddd; }
dt { font-weight: 600; } dd { margin: 0 0 .4rem; overflow-wrap: anywhere; }
.table { overflow-x: auto; margin-top: 1rem; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
td, th { padding: .35rem .55rem; border-bottom: 1px solid #d8dddd; text-align: left; }
thead { background: #e9efee; } tbody tr:nth-child(even) { background: #f0f4f3; }
.table td, .table th { white-space: nowrap; }
</style><body>
<h1>Industrial saved benchmark results</h1>
<p>Replay only. Cohorts use different splits, metrics or budgets. No joint ranking.
An incomplete search can have a saved score; it does not count as a successful pair.</p>
<p><a href="metrics.csv">Download metrics CSV</a></p>
'''


def write_saved_run_report(source_dir, output_dir):
    root = Path(source_dir)
    manifest = read_protocol_json(root / 'manifest.json')
    if manifest['version'] != 'industrial_data10_receipts@1' or manifest['visibility'] != 'public_aggregates':
        raise ProtocolError('Expected versioned public DATA-10 aggregate receipts')
    sources = manifest['sources']
    attempts = _read_receipt(root, sources['bd'], csv_file=True)
    stops = _stop_reasons(_read_receipt(root, sources['bd_stops'], csv_file=True))
    time = _read_receipt(root, sources['time'])
    nab = _read_receipt(root, sources['nab'])
    normalize = {'bd': lambda spec: _search_outcomes(spec, attempts, stops),
                 'time': lambda spec: _time_outcomes(spec, time), 'nab': lambda spec: _nab_outcomes(spec, nab)}

    tables, sections, links = [], [], []
    for spec in manifest['cohorts']:
        source_id = spec['source']
        rows = _report_rows(spec, normalize[source_id](spec))
        table = [(spec['id'], spec['task'], spec['metric'], row.cell.dataset, row.cell.model, row.cell.unit,
                  row.status, row.value, sources[source_id]['file'], row.artifact, row.reason) for row in rows]
        sections.append(_section(spec, rows, table, sources[source_id]))
        tables.extend(table)
        links.append(f'<a href="#{escape(spec["id"])}">{escape(spec["title"])}</a>')

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv_table(output / 'metrics.csv', _HEADERS, tables)
    (output / 'index.html').write_text(_PAGE + '<nav aria-label="Cohorts">' + ' '.join(links) + '</nav>\n'
                                       + '\n'.join(sections) + '</body></html>\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--source-dir', type=Path, default=_ROOT)
    args = parser.parse_args(argv)
    write_saved_run_report(args.source_dir, args.output_dir)


if __name__ == '__main__':
    main()
