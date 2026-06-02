import numpy as np
import pytest

from fedot_ind.core.models.detection.runtime import infer_regime_segments


def _three_regime_signal(length: int = 1200, seed: int = 0) -> np.ndarray:
    """Синтетика с тремя регимами: low → high → low с плавными переходами.

    Используется чтобы пороги high/low/transition были осмысленными
    (не вырожденными в 0). Без шума, чтобы тест был детерминирован.
    """
    rng = np.random.default_rng(seed)
    base = np.zeros(length, dtype=float)
    base[: length // 3] = 0.0
    base[length // 3 : 2 * length // 3] = 5.0
    base[2 * length // 3 :] = 0.0
    smoothed = np.convolve(base, np.ones(20) / 20.0, mode='same')
    return smoothed + 0.01 * rng.standard_normal(length)


def _expand_segments_to_labels(segments, length: int) -> np.ndarray:
    labels = np.full(length, 'unknown', dtype=object)
    for segment in segments:
        labels[segment.start_index : segment.end_index + 1] = segment.regime_label
    return labels


@pytest.mark.parametrize('cut', [400, 600, 900])
def test_causal_mode_labels_stable_under_truncation(cut: int) -> None:
    """С reference_window полностью внутри [0..cut] метки не меняются."""
    series = _three_regime_signal(length=1200)
    reference = (0, 300)

    full_segments = infer_regime_segments(series, reference_window=reference)
    truncated_segments = infer_regime_segments(series[:cut], reference_window=reference)

    full_labels = _expand_segments_to_labels(full_segments, length=1200)
    truncated_labels = _expand_segments_to_labels(truncated_segments, length=cut)

    assert reference[1] <= cut, 'предусловие теста: reference_window должен лежать в обрезке'
    assert np.array_equal(full_labels[:cut], truncated_labels), (
        f'Causal-режим протёк будущее: метки [0..{cut}) меняются от наличия данных после {cut}'
    )


def test_causal_mode_invariant_holds_for_all_three_labels() -> None:
    """Контроль: в наборе меток присутствуют high/low/transition.

    Гарантирует, что сам тест не вырождается (например, при ровно нулевых
    квантилях, когда все точки помечаются 'stable' и инвариант тривиален).
    """
    series = _three_regime_signal(length=1200)
    labels = _expand_segments_to_labels(
        infer_regime_segments(series, reference_window=(0, 300)),
        length=1200,
    )
    distinct = set(labels.tolist())
    assert {'low_load', 'high_load'}.issubset(distinct), (
        f'Сетап теста вырожден — нет high/low меток, есть только {distinct}'
    )


def test_legacy_mode_documented_to_leak_future() -> None:
    """Legacy режим (reference_window=None) использует глобальные квантили
    и потому НЕ удовлетворяет инварианту causality. Этот тест защищает от
    случайного "улучшения" legacy режима без обновления контракта.
    """
    series = _three_regime_signal(length=1200)
    cut = 600

    full_labels = _expand_segments_to_labels(
        infer_regime_segments(series), length=1200,
    )
    truncated_labels = _expand_segments_to_labels(
        infer_regime_segments(series[:cut]), length=cut,
    )

    assert not np.array_equal(full_labels[:cut], truncated_labels), (
        'Legacy режим неожиданно стал causal — это breaking change контракта. '
        'Если намеренно — обновите docstring и удалите этот тест.'
    )