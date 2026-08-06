"""Tests for fusion_over_raw Industrial preset, expander, and analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from benchmark.industrial import (
    RunStatus,
    TaskType,
    average_metrics_by_family,
    build_family_vs_best_single_frame,
    build_feature_combination_sensitivity_frame,
    build_fusion_over_raw_smoke_suite_config,
    build_fusion_over_raw_suite_config,
    build_minirocket_comparison_frame,
    expand_fusion_over_raw_model_specs,
    load_fusion_over_raw_defaults,
    render_fusion_over_raw_analysis_pack,
    run_local_benchmark_preset,
    run_tsc_benchmark_suite,
)
from benchmark.industrial.experiments.fusion_over_raw.config import (
    DEFAULTS_VERSION,
    FusionOverRawConfigError,
)
from benchmark.industrial.experiments.fusion_over_raw.expand import model_family_name


def test_load_fusion_over_raw_defaults() -> None:
    defaults = load_fusion_over_raw_defaults()

    assert defaults['version'] == DEFAULTS_VERSION
    assert defaults['primary_metric'] == 'f1_macro'
    assert 'f1_macro' in defaults['metrics']
    assert defaults['datasets'] == [
        'FordA',
        'ElectricDevices',
        'Crop',
        'StarLightCurves',
        'NonInvasiveFetalECGThorax1',
        'UWaveGestureLibraryAll',
    ]
    assert defaults['seeds'] == [42, 3407, 2025]


def test_build_fusion_over_raw_suite_config_uses_f1_macro_primary() -> None:
    config = build_fusion_over_raw_suite_config(persist_on_run=False)

    assert config.task_type is TaskType.TS_CLASSIFICATION
    assert config.run_spec.primary_metric == 'f1_macro'
    assert config.metrics[0:3] == ('accuracy', 'balanced_accuracy', 'f1_macro')
    assert [spec.dataset_name for spec in config.datasets] == [
        'FordA',
        'ElectricDevices',
        'Crop',
        'StarLightCurves',
        'NonInvasiveFetalECGThorax1',
        'UWaveGestureLibraryAll',
    ]


def test_expand_mvp_excludes_bottleneck_and_maps_raw_larger() -> None:
    models = expand_fusion_over_raw_model_specs(include_bottleneck=False)
    families = {model_family_name(spec.display_name) for spec in models}

    assert 'ordinary_bottleneck' not in ' '.join(families)
    assert not any('bottleneck' in family for family in families)
    assert 'raw_larger' in families
    assert 'MiniRocketRidge' in families

    raw_larger = next(spec for spec in models if model_family_name(spec.display_name) == 'raw_larger')
    assert raw_larger.params['modalities'] == ['raw']
    assert raw_larger.params['encoder_kwargs']['raw']['hidden_channels'] == [128, 192, 256]
    assert raw_larger.params['training']['validation_fraction'] == 0.2
    assert raw_larger.params['training']['learning_rate'] == 0.001
    assert raw_larger.params['fusion_kwargs'] == {'hidden_dim': 128, 'dropout': 0.2}

    fusion_concat = next(
        spec for spec in models if model_family_name(spec.display_name) == 'raw+stats__concat'
    )
    assert fusion_concat.params['fusion_method'] == 'concat'
    assert 'fusion_hidden_dim' not in fusion_concat.params['fusion_kwargs']

    # 5 baselines + 28 MVP fusions + 1 MiniRocket, times 3 seeds
    assert len(models) == (5 + 28 + 1) * 3


def test_expand_include_bottleneck_adds_variants() -> None:
    mvp = expand_fusion_over_raw_model_specs(include_bottleneck=False, seeds=(42,))
    full = expand_fusion_over_raw_model_specs(include_bottleneck=True, seeds=(42,))

    assert len(full) == len(mvp) + 3
    assert any('ordinary_bottleneck' in spec.display_name for spec in full)


def test_smoke_suite_config_shape() -> None:
    config = build_fusion_over_raw_smoke_suite_config(persist_on_run=False)

    assert len(config.datasets) == 1
    assert config.datasets[0].benchmark == 'in_memory_tsc'
    assert len(config.models) == 2
    assert config.run_spec.primary_metric == 'f1_macro'
    assert {model_family_name(spec.display_name) for spec in config.models} == {
        'raw',
        'raw+stats__concat',
    }
    seeds = {
        tag.split(':', 1)[1]
        for spec in config.models
        for tag in spec.tags
        if str(tag).startswith('seed:')
    }
    assert seeds == {'42'}


def test_run_fusion_over_raw_smoke_preset_selects_f1_macro(tmp_path: Path) -> None:
    result = run_local_benchmark_preset(
        'fusion_over_raw_smoke',
        output_dir=str(tmp_path / 'smoke'),
        persist_on_run=False,
    )

    assert result.config.task_type is TaskType.TS_CLASSIFICATION
    assert result.aggregate_report.primary_metric == 'f1_macro'
    assert any(record.status is RunStatus.SUCCESS for record in result.run_records)
    assert {record.model_name for record in result.run_records} >= {
        'raw__seed42',
        'raw+stats__concat__seed42',
    }


def test_run_fusion_over_raw_smoke_suite_direct(tmp_path: Path) -> None:
    config = build_fusion_over_raw_smoke_suite_config(
        output_dir=tmp_path / 'direct',
        persist_on_run=True,
    )
    result = run_tsc_benchmark_suite(config)

    assert result.aggregate_report.primary_metric == 'f1_macro'
    assert any(record.status is RunStatus.SUCCESS for record in result.run_records)
    metric_names = {record.metric_name for record in result.metric_records}
    assert 'f1_macro' in metric_names


def test_analysis_family_vs_best_single_and_minirocket(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            _row('D1', 'raw__seed42', 0.70),
            _row('D1', 'raw__seed3407', 0.72),
            _row('D1', 'stats__seed42', 0.60),
            _row('D1', 'raw+stats__concat__seed42', 0.80),
            _row('D1', 'raw+stats__gated__seed42', 0.78),
            _row('D1', 'MiniRocketRidge__seed42', 0.75),
            _row('D2', 'raw__seed42', 0.55),
            _row('D2', 'raw+stats__concat__seed42', 0.50),
            _row('D2', 'MiniRocketRidge__seed42', 0.65),
        ]
    )

    family = average_metrics_by_family(frame, metric_name='f1_macro')
    deltas = build_family_vs_best_single_frame(family, metric_name='f1_macro')
    minirocket = build_minirocket_comparison_frame(family, metric_name='f1_macro')
    sensitivity = build_feature_combination_sensitivity_frame(family, metric_name='f1_macro')
    manifest = render_fusion_over_raw_analysis_pack(
        frame,
        tmp_path / 'analysis',
        metric_name='f1_macro',
        expected_datasets=('D1', 'D2'),
    )

    raw_family = family[(family['dataset_name'] == 'D1') & (family['model_name'] == 'raw')]
    assert pytest.approx(float(raw_family['metric_value'].item()), rel=1e-6) == 0.71

    d1 = deltas[deltas['dataset_name'] == 'D1']
    concat = d1[d1['fusion_family'] == 'raw+stats__concat'].iloc[0]
    assert concat['best_single_model'] == 'raw'
    assert concat['improvement'] > 0

    d1_rocket = minirocket[minirocket['dataset_name'] == 'D1'].iloc[0]
    assert d1_rocket['target_model'] == 'raw+stats__concat'
    assert d1_rocket['improvement_vs_minirocket'] > 0

    assert not sensitivity.empty
    assert (tmp_path / 'analysis' / 'fusion_summary.md').is_file()
    assert any(record.format == 'csv' for record in manifest)


def test_unsupported_defaults_version(tmp_path: Path) -> None:
    path = tmp_path / 'bad.json'
    path.write_text('{"version": "wrong"}', encoding='utf-8')
    # Clear cache so a custom path still goes through validation.
    load_fusion_over_raw_defaults.cache_clear()
    with pytest.raises(FusionOverRawConfigError, match='Unsupported'):
        load_fusion_over_raw_defaults(path)
    load_fusion_over_raw_defaults.cache_clear()


def _row(dataset_name: str, model_name: str, metric_value: float) -> dict:
    return {
        'dataset_name': dataset_name,
        'model_name': model_name,
        'metric_name': 'f1_macro',
        'metric_value': metric_value,
        'source_label': 'unit',
        'task_type': 'ts_classification',
        'metric_direction': 'higher',
    }
