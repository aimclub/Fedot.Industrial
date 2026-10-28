"""Rendering effects for an already prepared per-series comparison."""
from pathlib import Path
import numpy as np
from benchmark.industrial.core import ArtifactRecord, RunStatus, ensure_directory, write_json


def render_series_comparison(result, series_id, dataset_name, series_predictions, metrics,
                             series_record, verbosity_policy, output_dir):
    from benchmark.industrial.evaluation.analytics import (
        _extract_forecasting_stage_payload, _extract_okhs_fdmd_stage_payload,
        _extract_stage_tuning_report, _intervals_from_mask, _is_hybrid_ensemble_record,
        _prune_generic_artifact_payload, _slugify_model_name,
    )
    artifact_manifest = []
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    target_dir = ensure_directory(output_dir)
    pivot = series_predictions.pivot(index='horizon_index', columns='model_name', values='y_pred')
    truth = (
        series_predictions[['horizon_index', 'y_true']]
        .drop_duplicates()
        .sort_values('horizon_index')
        .set_index('horizon_index')
    )
    train_values = np.asarray(series_record.train_values, dtype=float) if series_record is not None else np.array(
        [])
    actual_test = np.asarray(series_record.test_values, dtype=float) if series_record is not None else truth[
        'y_true'].to_numpy()
    forecast_index = np.arange(len(train_values), len(train_values) + len(actual_test))
    history_index = np.arange(len(train_values))
    zoom_width = max(len(actual_test) * 3, min(len(train_values), 48))
    zoom_start = max(0, len(train_values) - zoom_width)

    overlay_figure, overlay_axis = plt.subplots(figsize=(12, 6))
    if len(train_values):
        overlay_axis.plot(history_index, train_values, label='train', linewidth=2.0, color='0.45')
    overlay_axis.plot(forecast_index, actual_test, label='actual', linewidth=2.5, color='black')
    for model_name in pivot.columns:
        overlay_axis.plot(forecast_index, pivot[model_name].to_numpy(), label=model_name, linewidth=1.8)
    overlay_axis.axvline(len(train_values) - 1, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
    overlay_axis.set_title(f'History and Forecast Comparison for {series_id}')
    overlay_axis.set_xlabel('Time Index')
    overlay_axis.set_ylabel('Value')
    overlay_axis.legend(frameon=False)
    overlay_axis.grid(alpha=0.2)
    for extension in ('png', 'svg'):
        path = target_dir / f'{series_id}_history_forecast_overlay.{extension}'
        overlay_figure.savefig(path, dpi=200, bbox_inches='tight')
        artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
    plt.close(overlay_figure)

    boundary_figure, boundary_axis = plt.subplots(figsize=(12, 6))
    if len(train_values):
        boundary_axis.plot(
            np.arange(zoom_start, len(train_values)),
            train_values[zoom_start:],
            label='train',
            linewidth=2.0,
            color='0.45',
        )
    boundary_axis.plot(forecast_index, actual_test, label='actual', linewidth=2.5, color='black')
    for model_name in pivot.columns:
        boundary_axis.plot(forecast_index, pivot[model_name].to_numpy(), label=model_name, linewidth=1.8)
    boundary_axis.axvline(len(train_values) - 1, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
    boundary_axis.set_title(f'Boundary Zoom for {series_id}')
    boundary_axis.set_xlabel('Time Index')
    boundary_axis.set_ylabel('Value')
    boundary_axis.legend(frameon=False, ncol=2)
    boundary_axis.grid(alpha=0.2)
    for extension in ('png', 'svg'):
        path = target_dir / f'{series_id}_boundary_zoom.{extension}'
        boundary_figure.savefig(path, dpi=200, bbox_inches='tight')
        artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
    plt.close(boundary_figure)

    delta_figure, delta_axis = plt.subplots(figsize=(10, 5))
    last_train_value = float(train_values[-1]) if len(train_values) else 0.0
    labels = ['actual'] + list(pivot.columns)
    deltas = [float(actual_test[0] - last_train_value)] if len(actual_test) else [0.0]
    deltas.extend(float(pivot[model_name].iloc[0] - last_train_value) for model_name in pivot.columns)
    delta_axis.bar(labels, deltas, color=['black'] + [f'C{index % 10}' for index in range(len(pivot.columns))])
    delta_axis.axhline(0.0, color='0.4', linestyle='--', linewidth=1.0)
    delta_axis.set_title(f'First-Step Forecast Delta for {series_id}')
    delta_axis.set_xlabel('Series / Model')
    delta_axis.set_ylabel('Delta from Last Train Value')
    delta_axis.grid(alpha=0.2, axis='y')
    for extension in ('png', 'svg'):
        path = target_dir / f'{series_id}_forecast_delta.{extension}'
        delta_figure.savefig(path, dpi=200, bbox_inches='tight')
        artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
    plt.close(delta_figure)

    residual_figure, residual_axis = plt.subplots(figsize=(9, 5))
    for model_name in pivot.columns:
        residual_axis.plot(
            forecast_index,
            actual_test - pivot[model_name].to_numpy(),
            label=model_name,
            linewidth=1.6,
        )
    residual_axis.axhline(0.0, color='black', linestyle='--', linewidth=1)
    residual_axis.set_title(f'Residuals for {series_id}')
    residual_axis.set_xlabel('Horizon')
    residual_axis.set_ylabel('Residual')
    residual_axis.legend(frameon=False)
    residual_axis.grid(alpha=0.2)
    for extension in ('png', 'svg'):
        path = target_dir / f'{series_id}_residuals.{extension}'
        residual_figure.savefig(path, dpi=200, bbox_inches='tight')
        artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
    plt.close(residual_figure)

    horizon_metrics = metrics[
        (metrics['series_id'] == series_id)
        & (metrics['metric_name'] == result.aggregate_report.primary_metric)
        & (metrics['horizon_index'].notna())
    ].copy()
    if not horizon_metrics.empty:
        horizon_figure, horizon_axis = plt.subplots(figsize=(9, 5))
        for model_name, group in horizon_metrics.groupby('model_name'):
            horizon_axis.plot(group['horizon_index'], group['metric_value'], label=model_name, linewidth=1.8)
        horizon_axis.set_title(f'Horizon Error Profile for {series_id}')
        horizon_axis.set_xlabel('Horizon')
        horizon_axis.set_ylabel(result.aggregate_report.primary_metric.upper())
        horizon_axis.legend(frameon=False)
        horizon_axis.grid(alpha=0.2)
        for extension in ('png', 'svg'):
            path = target_dir / f'{series_id}_horizon_profile.{extension}'
            horizon_figure.savefig(path, dpi=200, bbox_inches='tight')
            artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
        plt.close(horizon_figure)

    series_run_records = [record for record in result.run_records if record.series_id == series_id]
    regime_reference = next(
        (record for record in series_run_records if record.metadata.get('regime_diagnostics')),
        None,
    )
    if regime_reference is not None:
        model_summaries = []
        for record in sorted(series_run_records, key=lambda item: item.model_name):
            model_summaries.append(
                {
                    'model_name': record.model_name,
                    'adapter_name': record.metadata.get('adapter_name'),
                    'status': record.status.value,
                    'primary_metric': record.metrics_summary.get(result.aggregate_report.primary_metric),
                }
            )
        regime_payload = {
            'series_id': series_id,
            'dataset_name': dataset_name,
            'primary_metric': result.aggregate_report.primary_metric,
            'regime_diagnostics': regime_reference.metadata.get('regime_diagnostics', {}),
            'routing_recommendation': regime_reference.metadata.get('routing_recommendation', {}),
            'model_outcomes': model_summaries,
        }
        diagnostics_path = target_dir / f'{series_id}_regime_diagnostics.json'
        write_json(diagnostics_path, _prune_generic_artifact_payload(regime_payload, verbosity_policy))
        artifact_manifest.append(ArtifactRecord(kind='structured', path=str(diagnostics_path), format='json'))

    stage_records = [
        record for record in series_run_records
        if record.status is RunStatus.SUCCESS and _extract_forecasting_stage_payload(record.metadata) is not None
    ]
    if stage_records:
        stage_payload = {}
        for record in stage_records:
            stage_payload[record.model_name] = {
                'adapter_name': record.metadata.get('adapter_name'),
                'model_adapter_family': record.metadata.get('model_adapter_family'),
                'routing_recommendation_family': record.metadata.get('routing_recommendation_family'),
                **(_extract_forecasting_stage_payload(record.metadata) or {}),
            }
        stage_path = target_dir / f'{series_id}_forecasting_stage_diagnostics.json'
        write_json(stage_path, _prune_generic_artifact_payload(stage_payload, verbosity_policy))
        artifact_manifest.append(ArtifactRecord(kind='structured', path=str(stage_path), format='json'))

    hybrid_records = [
        record for record in series_run_records
        if record.status is RunStatus.SUCCESS and _is_hybrid_ensemble_record(record.metadata)
    ]
    if hybrid_records:
        hybrid_payload = {
            record.model_name: _prune_generic_artifact_payload(record.metadata, verbosity_policy)
            for record in hybrid_records
        }
        hybrid_path = target_dir / f'{series_id}_hybrid_ensemble_diagnostics.json'
        write_json(hybrid_path, hybrid_payload)
        artifact_manifest.append(ArtifactRecord(kind='structured', path=str(hybrid_path), format='json'))

    stage_tuning_records = [
        record for record in series_run_records
        if record.status is RunStatus.SUCCESS and _extract_stage_tuning_report(record.metadata) is not None
    ]
    if stage_tuning_records:
        stage_tuning_payload = {
            record.model_name: {
                'adapter_name': record.metadata.get('adapter_name'),
                'model_adapter_family': record.metadata.get('model_adapter_family'),
                'routing_recommendation_family': record.metadata.get('routing_recommendation_family'),
                **(
                    verbosity_policy.prune_stage_tuning_report(
                        _extract_stage_tuning_report(record.metadata) or {}
                    ) or {}
                ),
            }
            for record in stage_tuning_records
        }
        stage_tuning_path = target_dir / f'{series_id}_forecasting_stage_tuning.json'
        write_json(stage_tuning_path, stage_tuning_payload)
        artifact_manifest.append(ArtifactRecord(kind='structured', path=str(stage_tuning_path), format='json'))

    okhs_records = [
        record for record in result.run_records
        if record.series_id == series_id and record.status is RunStatus.SUCCESS
        and 'okhs' in record.model_name.lower()
    ]
    if okhs_records:
        diagnostics_payload = {
            record.model_name: _prune_generic_artifact_payload(record.metadata, verbosity_policy)
            for record in okhs_records
        }
        diagnostics_path = target_dir / f'{series_id}_okhs_diagnostics.json'
        write_json(diagnostics_path, diagnostics_payload)
        artifact_manifest.append(ArtifactRecord(kind='structured', path=str(diagnostics_path), format='json'))

        okhs_fdmd_stage_payload = {}
        for record in okhs_records:
            stage_payload = _extract_okhs_fdmd_stage_payload(record.metadata)
            if stage_payload is None:
                continue
            okhs_fdmd_stage_payload[record.model_name] = {
                'adapter_name': record.metadata.get('adapter_name'),
                'model_adapter_family': record.metadata.get('model_adapter_family'),
                'routing_recommendation_family': record.metadata.get('routing_recommendation_family'),
                **stage_payload,
            }
        if okhs_fdmd_stage_payload:
            stage_diagnostics_path = target_dir / f'{series_id}_okhs_fdmd_stage_diagnostics.json'
            write_json(
                stage_diagnostics_path,
                _prune_generic_artifact_payload(okhs_fdmd_stage_payload, verbosity_policy),
            )
            artifact_manifest.append(
                ArtifactRecord(kind='structured', path=str(stage_diagnostics_path), format='json')
            )

        for record in okhs_records:
            fit_diagnostics = record.metadata.get('fdmd_fit_diagnostics', {})
            if not fit_diagnostics:
                continue
            eigen_real = np.asarray(fit_diagnostics.get('eigenvalues_real', []), dtype=float)
            eigen_imag = np.asarray(fit_diagnostics.get('eigenvalues_imag', []), dtype=float)
            mode_norms = np.asarray(fit_diagnostics.get('mode_norms', []), dtype=float)
            prediction_diagnostics = record.metadata.get('fdmd_prediction_diagnostics', {})
            model_slug = _slugify_model_name(record.model_name)

            mode_figure, mode_axes = plt.subplots(1, 2, figsize=(12, 5))
            if len(eigen_real):
                mode_axes[0].scatter(eigen_real, eigen_imag, alpha=0.85)
            mode_axes[0].axvline(0.0, color='0.5', linestyle='--', linewidth=1.0)
            mode_axes[0].set_title(f'Eigenvalues: {record.model_name}')
            mode_axes[0].set_xlabel('Re(lambda)')
            mode_axes[0].set_ylabel('Im(lambda)')
            mode_axes[0].grid(alpha=0.2)

            if len(mode_norms):
                mode_axes[1].bar(np.arange(len(mode_norms)), mode_norms)
            discontinuity = prediction_diagnostics.get('boundary_discontinuity_abs_mean')
            resolved_modes = fit_diagnostics.get('resolved_n_modes')
            anti_smoothing = prediction_diagnostics.get('anti_smoothing_diagnostics', {})
            collapse_detected = anti_smoothing.get('collapse_detected')
            envelope_ratio = anti_smoothing.get('envelope_ratio_before')
            mode_axes[1].set_title(
                f'Mode Norms (resolved={resolved_modes}, jump={discontinuity}, '
                f'collapse={collapse_detected}, env={envelope_ratio})'
            )
            mode_axes[1].set_xlabel('Mode Index')
            mode_axes[1].set_ylabel('Norm')
            mode_axes[1].grid(alpha=0.2, axis='y')

            for extension in ('png', 'svg'):
                path = target_dir / f'{series_id}_{model_slug}_okhs_modes.{extension}'
                mode_figure.savefig(path, dpi=200, bbox_inches='tight')
                artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
            plt.close(mode_figure)

    havok_records = [
        record for record in result.run_records
        if record.series_id == series_id and record.status is RunStatus.SUCCESS
        and 'havok' in record.model_name.lower()
    ]
    if havok_records:
        diagnostics_payload = {
            record.model_name: _prune_generic_artifact_payload(record.metadata, verbosity_policy)
            for record in havok_records
        }
        diagnostics_path = target_dir / f'{series_id}_havok_diagnostics.json'
        write_json(diagnostics_path, diagnostics_payload)
        artifact_manifest.append(ArtifactRecord(kind='structured', path=str(diagnostics_path), format='json'))

        for record in havok_records:
            model_slug = _slugify_model_name(record.model_name)
            forcing_values = np.asarray(record.metadata.get('forcing_values', []), dtype=float)
            forcing_threshold = float(record.metadata.get('forcing_threshold', 0.0))
            forcing_mask = np.asarray(record.metadata.get('forcing_mask', []), dtype=bool)
            forcing_intervals = record.metadata.get('forcing_active_intervals') or [
                [int(start), int(stop)] for start, stop in _intervals_from_mask(forcing_mask)
            ]
            forecast_forcing_values = np.asarray(record.metadata.get('forecast_forcing_values', []), dtype=float)
            forecast_forcing_mask = np.asarray(record.metadata.get('forecast_forcing_mask', []), dtype=bool)

            forcing_figure, forcing_axis = plt.subplots(figsize=(11, 4.5))
            if len(forcing_values):
                forcing_axis.plot(
                    np.arange(len(forcing_values)),
                    forcing_values,
                    linewidth=1.8,
                    color='C0',
                    label='forcing',
                )
            if forcing_threshold > 0:
                forcing_axis.axhline(forcing_threshold, color='C3', linestyle='--', linewidth=1.0)
                forcing_axis.axhline(-forcing_threshold, color='C3', linestyle='--', linewidth=1.0)
            for start, stop in forcing_intervals:
                forcing_axis.axvspan(start, stop + 1, color='C3', alpha=0.12)
            forcing_axis.set_title(f'HAVOK Forcing Timeline: {record.model_name}')
            forcing_axis.set_xlabel('Latent Step')
            forcing_axis.set_ylabel('Forcing')
            forcing_axis.grid(alpha=0.2)
            if len(forcing_values):
                forcing_axis.legend(frameon=False)
            for extension in ('png', 'svg'):
                path = target_dir / f'{series_id}_{model_slug}_havok_forcing_timeline.{extension}'
                forcing_figure.savefig(path, dpi=200, bbox_inches='tight')
                artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
            plt.close(forcing_figure)

            if record.model_name in pivot.columns:
                event_figure, event_axis = plt.subplots(figsize=(12, 6))
                if len(train_values):
                    event_axis.plot(history_index, train_values, label='train', linewidth=2.0, color='0.45')
                event_axis.plot(forecast_index, actual_test, label='actual', linewidth=2.5, color='black')
                event_axis.plot(
                    forecast_index,
                    pivot[record.model_name].to_numpy(),
                    label=record.model_name,
                    linewidth=1.8,
                    color='C0',
                )
                if len(forecast_forcing_mask):
                    for step_index, is_active in enumerate(forecast_forcing_mask[:len(forecast_index)]):
                        if is_active:
                            event_axis.axvspan(
                                forecast_index[step_index] - 0.5,
                                forecast_index[step_index] + 0.5,
                                color='C3',
                                alpha=0.12,
                            )
                event_axis.axvline(len(train_values) - 1, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
                event_axis.set_title(f'HAVOK Event Overlay for {series_id}')
                event_axis.set_xlabel('Time Index')
                event_axis.set_ylabel('Value')
                event_axis.legend(frameon=False)
                event_axis.grid(alpha=0.2)
                for extension in ('png', 'svg'):
                    path = target_dir / f'{series_id}_{model_slug}_havok_event_overlay.{extension}'
                    event_figure.savefig(path, dpi=200, bbox_inches='tight')
                    artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
                plt.close(event_figure)
    return artifact_manifest
