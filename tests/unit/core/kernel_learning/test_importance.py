import pytest

from fedot_ind.core.kernel_learning.contracts import KernelSelectionReport
from fedot_ind.core.kernel_learning.selection import KernelImportanceConfig, select_significant_generators


def _report(names, weights):
    return KernelSelectionReport(
        generator_names=tuple(names),
        weights=tuple(weights),
        selected_generators=tuple(name for name, weight in zip(names, weights) if weight > 0.0),
        selected_weights=tuple(float(weight) for weight in weights if weight > 0.0),
        scores={name: float(weight) for name, weight in zip(names, weights)},
        alignments={name: float(weight) for name, weight in zip(names, weights)},
        complexities={name: 0.0 for name in names},
        redundancies={name: 0.0 for name in names},
        task_type="classification",
    )


def test_select_significant_generators_uses_threshold_and_stable_weight_order():
    importance = select_significant_generators(
        _report(("low", "top_left", "top_right"), (0.01, 0.70, 0.70)),
        KernelImportanceConfig(weight_threshold=0.05),
    )

    assert importance.selected_generators == ("top_left", "top_right")
    assert importance.selected_weights == (0.70, 0.70)
    assert not importance.diagnostics["used_fallback"]


def test_select_significant_generators_falls_back_to_top_n_when_threshold_is_empty():
    importance = select_significant_generators(
        _report(("first", "second", "third"), (0.03, 0.02, 0.01)),
        KernelImportanceConfig(weight_threshold=0.50, fallback_top_n=2),
    )

    assert importance.selected_generators == ("first", "second")
    assert tuple(item.selected_by for item in importance.items) == ("fallback", "fallback")
    assert importance.diagnostics["used_fallback"]


def test_select_significant_generators_keeps_uniform_zero_score_weights_stable():
    importance = select_significant_generators(
        _report(("a", "b"), (0.5, 0.5)),
        KernelImportanceConfig(weight_threshold=0.05),
    )

    assert importance.selected_generators == ("a", "b")
    assert importance.selected_weights == (0.5, 0.5)


def test_kernel_importance_config_validates_bounds():
    with pytest.raises(ValueError):
        KernelImportanceConfig(weight_threshold=-0.1)
    with pytest.raises(ValueError):
        KernelImportanceConfig(fallback_top_n=0)
    with pytest.raises(ValueError):
        KernelImportanceConfig(max_union_size=0)
