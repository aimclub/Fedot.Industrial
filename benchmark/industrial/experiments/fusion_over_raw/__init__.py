"""Fusion-over-raw Industrial experiment preset and analysis helpers."""

from benchmark.industrial.experiments.fusion_over_raw.analysis import (
    average_metrics_by_family,
    build_bottleneck_variants_comparison_frame,
    build_family_vs_best_single_frame,
    build_feature_combination_sensitivity_frame,
    build_minirocket_comparison_frame,
    build_top_family_by_dataset_frame,
    classify_model_role,
    render_fusion_over_raw_analysis_pack,
)
from benchmark.industrial.experiments.fusion_over_raw.config import (
    BOTTLENECK_FUSIONS,
    DEFAULTS_PATH,
    DEFAULTS_VERSION,
    MVP_FUSIONS,
    FusionOverRawConfigError,
    load_fusion_over_raw_defaults,
)
from benchmark.industrial.experiments.fusion_over_raw.expand import (
    build_fusion_over_raw_smoke_suite_config,
    build_fusion_over_raw_suite_config,
    expand_fusion_over_raw_model_specs,
    model_family_name,
    parse_model_tags,
)

__all__ = [
    'BOTTLENECK_FUSIONS',
    'DEFAULTS_PATH',
    'DEFAULTS_VERSION',
    'FusionOverRawConfigError',
    'MVP_FUSIONS',
    'average_metrics_by_family',
    'build_bottleneck_variants_comparison_frame',
    'build_family_vs_best_single_frame',
    'build_feature_combination_sensitivity_frame',
    'build_fusion_over_raw_smoke_suite_config',
    'build_fusion_over_raw_suite_config',
    'build_minirocket_comparison_frame',
    'build_top_family_by_dataset_frame',
    'classify_model_role',
    'expand_fusion_over_raw_model_specs',
    'load_fusion_over_raw_defaults',
    'model_family_name',
    'parse_model_tags',
    'render_fusion_over_raw_analysis_pack',
]
