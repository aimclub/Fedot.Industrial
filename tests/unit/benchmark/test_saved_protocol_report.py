import csv
from collections import Counter
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil

import pytest

from benchmark.industrial.examples.report_saved_runs import Cell, Outcome, _with_coverage, main
from benchmark.industrial.protocols.schema import ProtocolError


_SOURCE = Path(__file__).resolve().parents[3] / 'benchmark/industrial/examples/saved_protocol_report'


def _csv(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def _fixture_sources(tmp_path):
    return shutil.copytree(_SOURCE, tmp_path / 'sources', ignore=shutil.ignore_patterns('report'))


def _replace_receipt(root, receipt, content):
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    source = manifest['sources'][receipt]
    (root / source['file']).write_bytes(content)
    source['sha256'] = sha256(content).hexdigest()
    path.write_text(json.dumps(manifest), encoding='utf-8')


class MetricCells(HTMLParser):
    def __init__(self):
        super().__init__()
        self.metrics = False
        self.cell = None
        self.values = []

    def handle_starttag(self, tag, attrs):
        if tag == 'div' and ('class', 'table') in attrs:
            self.metrics = True
        if self.metrics and tag == 'td':
            self.cell = ''

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if tag == 'td' and self.cell is not None:
            self.values.append(self.cell)
            self.cell = None
        if tag == 'div':
            self.metrics = False


def test_saved_report_replays_actual_receipts_and_matches_html(tmp_path):
    main([str(tmp_path)])
    rows = _csv(tmp_path / 'metrics.csv')
    assert len(rows) == 161
    assert Counter((row['task'], row['status']) for row in rows) == {
        ('ts_classification', 'success'): 48, ('ts_classification', 'incomplete'): 1,
        ('ts_classification', 'failed'): 1,
        ('ts_regression', 'success'): 25, ('ts_regression', 'incomplete'): 22, ('ts_regression', 'failed'): 3,
        ('forecasting', 'success'): 60,
        ('anomaly_detection', 'success'): 1,
    }
    by_artifact = {row['artifact']: row for row in rows if row['source'] == 'bd_attempts.csv'}
    attempts = _csv(_SOURCE / 'bd_attempts.csv')
    assert len(by_artifact) == len(attempts) == 150
    for attempt in attempts:
        row = by_artifact[attempt['path']]
        assert row['value'] == attempt['control_score']
        assert row['unit'] == attempt['seed']
        assert bool(row['reason']) == (attempt['passed'] == 'False')
    time = json.loads((_SOURCE / 'time_acceptance.json').read_text(encoding='utf-8'))
    by_window = {(row['model'], row['unit']): float(row['value']) for row in rows if row['cohort'] == 'time-forecast'}
    assert by_window == {(model, window['origin']): window[field] for window in time['windows']
                         for model, field in (('chronos-bolt-tiny', 'mae'), ('last_value@1', 'last_value_mae'))}
    assert next(float(row['value']) for row in rows if row['cohort'] == 'nab-detection') == 53 / 6766
    html = (tmp_path / 'index.html').read_text(encoding='utf-8')
    cells = MetricCells()
    cells.feed(html)
    assert cells.values == [value for row in rows for value in row.values()]
    assert rows == _csv(_SOURCE / 'report/metrics.csv')
    assert html == (_SOURCE / 'report/index.html').read_text(encoding='utf-8')


def test_missing_receipts_remain_in_coverage_but_tampered_receipts_fail(tmp_path):
    shutil.copyfile(_SOURCE / 'manifest.json', tmp_path / 'manifest.json')
    out = tmp_path / 'out'
    main([str(out), '--source-dir', str(tmp_path)])
    rows = _csv(out / 'metrics.csv')
    assert len(rows) == 161
    assert {(row['status'], row['value'], row['reason']) for row in rows} == {
        ('unavailable', '', 'no_saved_outcome'),
    }
    (tmp_path / 'bd_attempts.csv').write_text('not a receipt', encoding='utf-8')
    with pytest.raises(ProtocolError, match=r'checksum.*bd_attempts\.csv'):
        main([str(tmp_path / 'invalid'), '--source-dir', str(tmp_path)])
    assert not (tmp_path / 'invalid').exists()


def test_saved_report_checks_time_release_even_with_matching_checksum(tmp_path):
    root = _fixture_sources(tmp_path)
    time = json.loads((root / 'time_acceptance.json').read_text(encoding='utf-8'))
    time['dataset_release'] = 'another-release'
    _replace_receipt(root, 'time', json.dumps(time).encode('utf-8'))
    out = tmp_path / 'out'
    with pytest.raises(ProtocolError, match='TIME receipt.*dataset release'):
        main([str(out), '--source-dir', str(root)])
    assert not out.exists()


def test_saved_report_parses_the_verified_json_snapshot(tmp_path, monkeypatch):
    root = _fixture_sources(tmp_path)
    receipt = root / 'time_acceptance.json'
    read_bytes = Path.read_bytes

    def replace_after_read(path):
        data = read_bytes(path)
        if path == receipt:
            changed = json.loads(data)
            changed['windows'][0]['mae'] += 1
            path.write_text(json.dumps(changed), encoding='utf-8')
        return data

    monkeypatch.setattr(Path, 'read_bytes', replace_after_read)
    target = tmp_path / 'out'
    main([str(target), '--source-dir', str(root)])
    assert _csv(target / 'metrics.csv') == _csv(_SOURCE / 'report/metrics.csv')


def test_saved_report_rejects_conflicting_stop_reasons(tmp_path):
    root = _fixture_sources(tmp_path)
    data = (root / 'bd_stop_reasons.csv').read_bytes().splitlines()
    duplicate = data[1].replace(b'next_population_under_deadline', b'another_cause')
    assert duplicate != data[1]
    _replace_receipt(root, 'bd_stops', b'\n'.join([*data, duplicate]) + b'\n')
    out = tmp_path / 'out'
    with pytest.raises(ProtocolError, match='Conflicting stop reasons.*25'):
        main([str(out), '--source-dir', str(root)])
    assert not out.exists()


def test_replayed_cell_is_counted_once_and_conflicting_or_foreign_cells_fail():
    cell = Cell('one', 'B', '42')
    observed = Outcome(cell, 'success', .75, 'path')
    assert _with_coverage((cell,), (observed, observed)) == (observed,)
    with pytest.raises(ProtocolError, match='Conflicting'):
        _with_coverage((cell,), (observed, Outcome(cell, 'success', .5, 'path')))
    with pytest.raises(ProtocolError, match='outside'):
        _with_coverage((), (observed,))
    with pytest.raises(ProtocolError, match='Non-finite'):
        Outcome(cell, 'success', float('nan'), 'path')
