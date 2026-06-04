from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmark.industrial.core import ArtifactRecord, ForecastingBenchmarkResult, RunStatus, ensure_directory, \
    to_plain_data, write_json

DEFAULT_OKHS_SMOOTHING_SERIES_IDS = ('D364', 'D377', 'D378')


@dataclass(frozen=True)
class OKHSSmoothingSeriesSummary:
    series_id: str
    model_name: str
    status: str
    collapse_detected: bool
    correction_applied: bool
    collapse_resolved: bool
    mae: float | None
    smape: float | None
    mase: float | None
    envelope_ratio_before: float | None
    envelope_ratio_after: float | None
    forecast_amplitude_before: float | None
    forecast_amplitude_after: float | None


@dataclass(frozen=True)
class OKHSSmoothingSummary:
    model_name: str
    series_count: int
    success_count: int
    collapse_count: int
    corrected_count: int
    resolved_count: int
    collapse_rate: float
    corrected_rate: float
    resolved_rate: float
    mean_mae: float | None
    mean_smape: float | None
    mean_mase: float | None
    mean_envelope_ratio_before: float | None
    mean_envelope_ratio_after: float | None
    mean_amplitude_gain: float | None
    rows: tuple[OKHSSmoothingSeriesSummary, ...]


@dataclass(frozen=True)
class OKHSSmoothingAcceptanceCriteria:
    max_collapse_rate: float = 1.0
    min_resolved_rate: float = 0.5
    min_corrected_rate: float = 0.5
    min_mean_amplitude_gain: float = 0.0
    max_mean_envelope_ratio_after: float | None = 0.8


@dataclass(frozen=True)
class OKHSSmoothingAcceptanceReport:
    passed: bool
    criteria: OKHSSmoothingAcceptanceCriteria
    summary: OKHSSmoothingSummary
    reasons: tuple[str, ...]


def summarize_okhs_smoothing_result(
        result: ForecastingBenchmarkResult,
        *,
        model_name: str = 'OKHS DMD',
) -> OKHSSmoothingSummary:
    rows: list[OKHSSmoothingSeriesSummary] = []
    for record in result.run_records:
        if record.model_name != model_name:
            continue
        anti_smoothing = (
            record.metadata
            .get('fdmd_prediction_diagnostics', {})
            .get('anti_smoothing_diagnostics', {})
        )
        rows.append(
            OKHSSmoothingSeriesSummary(
                series_id=record.series_id,
                model_name=record.model_name,
                status=record.status.value,
                collapse_detected=bool(anti_smoothing.get('collapse_detected', False)),
                correction_applied=bool(anti_smoothing.get('correction_applied', False)),
                collapse_resolved=bool(anti_smoothing.get('collapse_resolved', False)),
                mae=record.metrics_summary.get('mae'),
                smape=record.metrics_summary.get('smape'),
                mase=record.metrics_summary.get('mase'),
                envelope_ratio_before=_maybe_float(anti_smoothing.get('envelope_ratio_before')),
                envelope_ratio_after=_maybe_float(anti_smoothing.get('envelope_ratio_after')),
                forecast_amplitude_before=_maybe_float(anti_smoothing.get('forecast_amplitude_before')),
                forecast_amplitude_after=_maybe_float(anti_smoothing.get('forecast_amplitude_after')),
            )
        )

    if not rows:
        raise ValueError(f'No OKHS smoothing records found for model_name={model_name}.')

    success_rows = [row for row in rows if row.status == RunStatus.SUCCESS.value]
    collapse_count = sum(1 for row in success_rows if row.collapse_detected)
    corrected_count = sum(1 for row in success_rows if row.correction_applied)
    resolved_count = sum(1 for row in success_rows if row.collapse_resolved)
    mean_amplitude_gain = _safe_mean(
        [
            (row.forecast_amplitude_after - row.forecast_amplitude_before)
            for row in success_rows
            if row.forecast_amplitude_after is not None and row.forecast_amplitude_before is not None
        ]
    )
    return OKHSSmoothingSummary(
        model_name=model_name,
        series_count=len(rows),
        success_count=len(success_rows),
        collapse_count=collapse_count,
        corrected_count=corrected_count,
        resolved_count=resolved_count,
        collapse_rate=float(collapse_count / max(1, len(success_rows))),
        corrected_rate=float(corrected_count / max(1, len(success_rows))),
        resolved_rate=float(resolved_count / max(1, len(success_rows))),
        mean_mae=_safe_mean([row.mae for row in success_rows if row.mae is not None]),
        mean_smape=_safe_mean([row.smape for row in success_rows if row.smape is not None]),
        mean_mase=_safe_mean([row.mase for row in success_rows if row.mase is not None]),
        mean_envelope_ratio_before=_safe_mean(
            [row.envelope_ratio_before for row in success_rows if row.envelope_ratio_before is not None]
        ),
        mean_envelope_ratio_after=_safe_mean(
            [row.envelope_ratio_after for row in success_rows if row.envelope_ratio_after is not None]
        ),
        mean_amplitude_gain=mean_amplitude_gain,
        rows=tuple(rows),
    )


def evaluate_okhs_smoothing_acceptance(
        summary: OKHSSmoothingSummary,
        criteria: OKHSSmoothingAcceptanceCriteria | None = None,
) -> OKHSSmoothingAcceptanceReport:
    resolved_criteria = criteria or OKHSSmoothingAcceptanceCriteria()
    reasons: list[str] = []
    if summary.collapse_rate > resolved_criteria.max_collapse_rate:
        reasons.append(
            f'collapse_rate={summary.collapse_rate:.3f} exceeds max_collapse_rate={resolved_criteria.max_collapse_rate:.3f}'
        )
    if summary.corrected_rate < resolved_criteria.min_corrected_rate:
        reasons.append(
            f'corrected_rate={summary.corrected_rate:.3f} is below min_corrected_rate={resolved_criteria.min_corrected_rate:.3f}'
        )
    if summary.resolved_rate < resolved_criteria.min_resolved_rate:
        reasons.append(
            f'resolved_rate={summary.resolved_rate:.3f} is below min_resolved_rate={resolved_criteria.min_resolved_rate:.3f}'
        )
    amplitude_gain = summary.mean_amplitude_gain if summary.mean_amplitude_gain is not None else float('-inf')
    if amplitude_gain < resolved_criteria.min_mean_amplitude_gain:
        reasons.append(
            f'mean_amplitude_gain={amplitude_gain:.3f} is below min_mean_amplitude_gain={resolved_criteria.min_mean_amplitude_gain:.3f}'
        )
    if (
            resolved_criteria.max_mean_envelope_ratio_after is not None
            and summary.mean_envelope_ratio_after is not None
            and summary.mean_envelope_ratio_after > resolved_criteria.max_mean_envelope_ratio_after
    ):
        reasons.append(
            'mean_envelope_ratio_after='
            f'{summary.mean_envelope_ratio_after:.3f} exceeds '
            f'max_mean_envelope_ratio_after={resolved_criteria.max_mean_envelope_ratio_after:.3f}'
        )
    return OKHSSmoothingAcceptanceReport(
        passed=not reasons,
        criteria=resolved_criteria,
        summary=summary,
        reasons=tuple(reasons),
    )


def has_okhs_smoothing_diagnostics(
        result: ForecastingBenchmarkResult,
        *,
        model_name: str = 'OKHS DMD',
) -> bool:
    for record in result.run_records:
        if record.model_name != model_name:
            continue
        anti_smoothing = (
            record.metadata
            .get('fdmd_prediction_diagnostics', {})
            .get('anti_smoothing_diagnostics')
        )
        if anti_smoothing:
            return True
    return False


def render_okhs_smoothing_acceptance_pack(
        result: ForecastingBenchmarkResult,
        output_dir: str | Path,
        *,
        model_name: str = 'OKHS DMD',
        criteria: OKHSSmoothingAcceptanceCriteria | None = None,
) -> tuple[ArtifactRecord, ...]:
    if not has_okhs_smoothing_diagnostics(result, model_name=model_name):
        return ()

    target_dir = ensure_directory(output_dir)
    summary = summarize_okhs_smoothing_result(result, model_name=model_name)
    report = evaluate_okhs_smoothing_acceptance(summary, criteria)

    artifacts: list[ArtifactRecord] = []
    summary_path = Path(target_dir) / 'okhs_smoothing_summary.json'
    write_json(summary_path, to_plain_data(summary))
    artifacts.append(ArtifactRecord(kind='structured', path=str(summary_path), format='json'))

    acceptance_path = Path(target_dir) / 'okhs_smoothing_acceptance.json'
    write_json(acceptance_path, to_plain_data(report))
    artifacts.append(ArtifactRecord(kind='structured', path=str(acceptance_path), format='json'))

    markdown_path = Path(target_dir) / 'okhs_smoothing_acceptance.md'
    markdown_lines = [
        '# OKHS Smoothing Acceptance Report',
        '',
        f'- Model: `{summary.model_name}`',
        f'- Passed: `{report.passed}`',
        f'- Series count: `{summary.series_count}`',
        f'- Success count: `{summary.success_count}`',
        f'- Collapse rate: `{summary.collapse_rate:.3f}`',
        f'- Corrected rate: `{summary.corrected_rate:.3f}`',
        f'- Resolved rate: `{summary.resolved_rate:.3f}`',
        f'- Mean amplitude gain: `{summary.mean_amplitude_gain if summary.mean_amplitude_gain is not None else "n/a"}`',
        '',
        '## Acceptance Criteria',
        '',
        f'- `max_collapse_rate = {report.criteria.max_collapse_rate}`',
        f'- `min_corrected_rate = {report.criteria.min_corrected_rate}`',
        f'- `min_resolved_rate = {report.criteria.min_resolved_rate}`',
        f'- `min_mean_amplitude_gain = {report.criteria.min_mean_amplitude_gain}`',
        f'- `max_mean_envelope_ratio_after = {report.criteria.max_mean_envelope_ratio_after}`',
        '',
        '## Reasons',
        '',
    ]
    if report.reasons:
        markdown_lines.extend(f'- {reason}' for reason in report.reasons)
    else:
        markdown_lines.append('- acceptance passed')
    markdown_lines.extend(
        ['', '## Series', '',
         '| series_id | status | collapse_detected | correction_applied | collapse_resolved | mae | envelope_ratio_before | envelope_ratio_after |',
         '|---|---|---:|---:|---:|---:|---:|---:|', ])
    for row in summary.rows:
        markdown_lines.append(
            f'| {row.series_id} | {row.status} | {row.collapse_detected} | {row.correction_applied} | '
            f'{row.collapse_resolved} | {row.mae} | {row.envelope_ratio_before} | {row.envelope_ratio_after} |'
        )
    markdown_path.write_text('\n'.join(markdown_lines), encoding='utf-8')
    artifacts.append(ArtifactRecord(kind='summary', path=str(markdown_path), format='md'))

    return tuple(artifacts)


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _maybe_float(value) -> float | None:
    if value is None:
        return None
    return float(value)
