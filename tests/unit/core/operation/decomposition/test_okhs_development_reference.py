from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[5]
    / "docs"
    / "okhs_dmd"
    / "okhs_development_reference.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("okhs_development_reference", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _unwrap_left_message(either_result):
    return either_result.either(lambda value: value, lambda value: value)


def test_build_analysis_report_contains_expected_sections():
    module = _load_module()

    report = module.build_analysis_report()
    markdown = module.render_markdown_report(report)

    assert "OKHS-DMD Alignment Analysis" in markdown
    assert "Alignment Matrix" in markdown
    assert "Train/test leakage" in markdown
    assert "pymittagleffler" in markdown


def test_render_markdown_report_is_deterministic():
    module = _load_module()

    report = module.build_analysis_report()
    first = module.render_markdown_report(report)
    second = module.render_markdown_report(report)

    assert first == second


def test_choose_holdout_split_has_no_overlap_and_expected_sizes():
    module = _load_module()

    result = module.choose_holdout_split(total_trajectories=6, holdout_size=2)

    assert result.is_right()
    split = result.value
    assert split.train_indices == (0, 1, 2, 3)
    assert split.test_indices == (4, 5)
    assert set(split.train_indices).isdisjoint(split.test_indices)


def test_validate_initial_segment_length_rejects_insufficient_observations():
    module = _load_module()

    result = module.validate_initial_segment_length(
        initial_segment_length=3,
        n_modes=8,
        n_features=2,
    )

    assert result.is_left()
    assert "Insufficient initial segment length" in _unwrap_left_message(result)


def test_validate_liouville_shapes_accepts_consistent_shapes():
    module = _load_module()

    result = module.validate_liouville_shapes(
        gram_shape=(4, 4),
        liouville_shape=(4, 4),
        eigen_count=4,
    )

    assert result.is_right()


def test_validate_liouville_shapes_rejects_mismatch():
    module = _load_module()

    result = module.validate_liouville_shapes(
        gram_shape=(4, 4),
        liouville_shape=(3, 3),
        eigen_count=2,
    )

    assert result.is_left()
    assert "Gram/Liouville mismatch" in _unwrap_left_message(result)


def test_normalize_okhs_method_supports_expected_aliases():
    module = _load_module()

    direct = module.normalize_okhs_method("direct")
    dmd = module.normalize_okhs_method("dmd")
    occupation = module.normalize_okhs_method("occupation")

    assert direct.is_right()
    assert dmd.is_right()
    assert occupation.is_right()
    assert direct.value == module.OKHSMethod.DIRECT
    assert dmd.value == module.OKHSMethod.DMD
    assert occupation.value == module.OKHSMethod.OCCUPATION


def test_detect_gaps_reports_undeclared_dependency():
    module = _load_module()

    report = module.build_analysis_report()
    gaps = module.detect_gaps(report.alignments, report.dependency_gaps)

    assert "undeclared dependency: pymittagleffler" in gaps
