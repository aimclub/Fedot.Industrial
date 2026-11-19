"""Build one shareable report from saved ROAST metric records and a resolved plan."""

from dataclasses import dataclass, replace
import csv
import html
import json
from pathlib import Path

from ..protocols.schema import (
    EvaluationMode, MembershipSplit, MetricDirection, Partition, ProtocolError, fields, integer, mapping,
    string, unique_strings,
)


@dataclass(frozen=True)
class ReportRow:
    model: str
    dataset: str
    item: str
    repeat: int
    status: str
    value: float | None
    artifact: str
    reason: str


@dataclass(frozen=True)
class ProtocolReport:
    comparison_key: str
    plan_id: str
    metric: str
    repeat_count: int
    rows: tuple[ReportRow, ...]
    ranked: tuple[tuple[str, float], ...]
    context: dict

    @property
    def expected(self):
        return len(self.rows)

    @property
    def successful(self):
        return sum(row.status == 'success' for row in self.rows)


def _metric_rule(plan, metric):
    candidates = [dict(profile.protocol.metrics).get(metric) for profile in plan.profiles]
    if not candidates or candidates[0] is None or any(rule != candidates[0] for rule in candidates[1:]):
        raise ProtocolError('Cross-dataset report needs matching metric version, direction and aggregation')
    return candidates[0]


def _evaluation_items(profile):
    protocol = profile.protocol
    split = protocol.split
    selected = Partition.TEST if protocol.mode is EvaluationMode.OFFICIAL else Partition.VALIDATION
    if isinstance(split, MembershipSplit):
        return getattr(split, selected.value)
    return tuple(case.item_id for case in split.cases if case.partition is selected)


def _record_index(plan, records, allowed, metric):
    indexed, seen, origins = {}, {}, {}
    for record in records:
        if record.metric_id != metric:
            raise ProtocolError('A report must select exactly one metric; select metrics before reporting')
        info = record.to_dict()
        if seen.setdefault(record.record_id, info) != info:
            raise ProtocolError('A repeated record_id has conflicting contents')
        meta = record.metadata
        if (meta.get('plan_id') != plan.plan_id or meta.get('comparison_key') != plan.comparison_key
                or meta.get('source') != 'local_run'):
            raise ProtocolError('Records from another plan or external publication are not ranked together')
        repeat = integer(record.coordinates.get('repeat'), 'record repeat', minimum=0)
        key = (record.model_id, record.dataset_id, record.item_id, repeat)
        if key not in allowed:
            raise ProtocolError('Record falls outside the declared coverage grid')
        if origins.setdefault(key, record.record_id) != record.record_id:
            raise ProtocolError('Conflicting metric records for the same evaluation cell')
        if record.status.value == 'success' and record.value is None:
            raise ProtocolError('Successful metric records need values')
        artifact = meta.get('artifact_ref')
        artifact = string(artifact, 'relative artifact reference')
        parts = artifact.split('/')
        if any(part in ('', '.', '..') for part in parts) or any(char in artifact for char in ':\\\n\r'):
            raise ProtocolError('Artifact references must use relative, portable paths')
        if record.status.value != 'success' and record.value is not None:
            raise ProtocolError('Unsuccessful metric records must not carry a score')
        row = ReportRow(*key, record.status.value, record.value, artifact, str(meta.get('reason', '')))
        indexed[key] = row
    return indexed


def build_protocol_report(plan, records, *, items_by_dataset, repeat_count):
    """Use explicit coverage units; absent or failed cells prevent ranking."""
    integer(repeat_count, 'repeat_count')
    metric = plan.suite.run.primary_metric
    rule = _metric_rule(plan, metric)
    if set(items_by_dataset) != {profile.dataset_id for profile in plan.profiles}:
        raise ProtocolError('Report coverage must describe all datasets in the plan')
    units = {key: unique_strings(items, f'{key} report units') for key, items in items_by_dataset.items()}
    if any(not items for items in units.values()):
        raise ProtocolError('Each report dataset must declare evaluation units')
    allowed = {(spec.model_id, dataset, item, repeat)
               for spec in plan.suite.models for dataset, items in units.items()
               for item in items for repeat in range(repeat_count)}
    indexed = _record_index(plan, records, allowed, metric)
    rows = tuple(indexed.get(key, ReportRow(*key, 'unavailable', None, '', 'no saved outcome'))
                 for key in sorted(allowed))
    ranked = ()
    modes = {profile.protocol.mode for profile in plan.profiles}
    full_coverage = all(set(units[profile.dataset_id]) == set(_evaluation_items(profile)) for profile in plan.profiles)
    # Only the explicit mean-of-units rule produces a cross-dataset scalar here.
    if full_coverage and modes == {EvaluationMode.OFFICIAL} and rule.aggregation == 'mean_by_unit' and all(
        row.status == 'success' for row in rows
    ):
        by_model = {}
        for row in rows:
            by_model.setdefault(row.model, []).append(row.value)
        direction = -1 if rule.direction is MetricDirection.MAXIMIZE else 1
        ranked = tuple(sorted(((model, sum(values) / len(values)) for model, values in by_model.items()),
                              key=lambda item: (direction * item[1], item[0])))
    context = {'task': plan.suite.task_kind, 'budget': dict(plan.suite.run.options['evaluation_budget']),
               'all_planned_units': full_coverage,
               'metric_rule': {'version': rule.version, 'direction': rule.direction.value,
                               'aggregation': rule.aggregation},
               'profiles': {profile.dataset_id: {
                   'release': profile.release.version, 'sha256': profile.release.sha256,
                   'protocol': profile.protocol.name, 'version': profile.protocol.version,
                   'mode': profile.protocol.mode.value, 'split_kind': profile.protocol.split.kind,
                   'expected_units': len(_evaluation_items(profile)),
               } for profile in plan.profiles}}
    return ProtocolReport(plan.comparison_key, plan.plan_id, metric, repeat_count, rows, ranked, context)


def report_from_evidence(plan, payload):
    """Deserialize saved native ROAST records from an explicitly public envelope."""
    from roast.core.records import MetricRecord
    from roast.core.schema import SchemaError

    evidence = fields(payload, ('version', 'visibility', 'items_by_dataset', 'repeat_count',
                                'metrics', 'provenance'), 'report evidence')
    if evidence['version'] != 'industrial_report_evidence@1' or evidence['visibility'] != 'public_aggregates':
        raise ProtocolError('Report evidence must declare a supported version and public aggregate units')
    provenance = fields(evidence['provenance'], ('code_revision', 'recorded_at'), 'report provenance')
    if not isinstance(evidence['metrics'], list):
        raise ProtocolError('Report metric records must be a JSON list')
    try:
        records = [MetricRecord.from_dict(mapping(record, 'metric record')) for record in evidence['metrics']]
    except SchemaError as error:
        raise ProtocolError(f'Metric record: {error}') from error
    report = build_protocol_report(plan, records,
                                   items_by_dataset=mapping(evidence['items_by_dataset'], 'report units'),
                                   repeat_count=evidence['repeat_count'])
    return replace(report, context={**report.context,
                                    **{key: string(value, key) for key, value in provenance.items()}})


def pairwise_units(report: ProtocolReport, left, right):
    """Show matched cells and retain all unmatched coordinates as a coverage gap."""
    def successful(model):
        return {(row.dataset, row.item, row.repeat): row.value for row in report.rows
                if row.model == model and row.status == 'success'}

    models = {row.model for row in report.rows}
    if left == right or {left, right} - models:
        raise ProtocolError('Select two different models from the same report')
    first, second = successful(left), successful(right)
    matched = first.keys() & second.keys()
    return {'left': left, 'right': right, 'unit': 'dataset/item', 'declared_repeats': report.repeat_count,
            'matched': tuple((*key, first[key] - second[key]) for key in sorted(matched)),
            'left_only': tuple(sorted(first.keys() - matched)), 'right_only': tuple(sorted(second.keys() - matched)),
            'neither': tuple((row.dataset, row.item, row.repeat) for row in report.rows if row.model == left
                             and (row.dataset, row.item, row.repeat) not in first
                             and (row.dataset, row.item, row.repeat) not in second)}


_HEADERS = ('model', 'dataset', 'item', 'repeat', 'status', 'value', 'artifact', 'reason')


def write_csv_table(path, headers, table):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(table)


def render_html_table(headers, table):
    def cell(value):
        return html.escape('' if value is None else str(value))

    head = ''.join(f'<th scope="col">{cell(value)}</th>' for value in headers)
    body = '\n'.join('<tr>' + ''.join(f'<td>{cell(value)}</td>' for value in row) + '</tr>' for row in table)
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def write_protocol_report(report: ProtocolReport, root, *, inputs=()):
    """Render CSV and HTML from the same rows; no evaluator is invoked."""
    root = Path(root)
    inputs = tuple(map(Path, inputs))
    for target in (root / name for name in ('metrics.csv', 'summary.json', 'index.html')):
        for source in inputs:
            if (target.resolve() == source.resolve()
                    or (target.exists() and source.exists() and target.samefile(source))):
                raise ProtocolError(f'Report output would overwrite an input: {target.name}')
    root.mkdir(parents=True, exist_ok=True)
    table = tuple(tuple(getattr(row, column) for column in _HEADERS) for row in report.rows)
    write_csv_table(root / 'metrics.csv', _HEADERS, table)
    info = {'plan_id': report.plan_id, 'comparison_key': report.comparison_key, 'metric': report.metric,
            'coverage': {'success': report.successful, 'expected': report.expected},
            'repeat_count': report.repeat_count, 'ranked': report.ranked, 'context': report.context}
    (root / 'summary.json').write_text(json.dumps(info, indent=2, allow_nan=False), encoding='utf-8')
    summary = html.escape(json.dumps(info, indent=2, allow_nan=False))
    (root / 'index.html').write_text(
        f'<!doctype html><meta charset="utf-8"><title>Benchmark protocol report</title>'
        f'<h1>Benchmark protocol report</h1><pre>{summary}</pre>{render_html_table(_HEADERS, table)}',
        encoding='utf-8')
