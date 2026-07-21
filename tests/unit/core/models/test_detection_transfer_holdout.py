import numpy as np
import pytest

from fedot_ind.core.models.detection.runtime import (
    DetectionSplitKind,
    DetectionSplitSpec,
)
from fedot_ind.core.models.detection.stage_tuning_runtime import (
    _split_series,
    iter_domain_holdouts,
    run_detection_stage_tuning_on_series,
)


def _three_domain_series(per_domain: int = 4):
    """Синтетика из 3 доменов (A, B, C) с разными средними уровнями."""
    means = {'A': 0.0, 'B': 10.0, 'C': 20.0}
    values_parts = []
    domain_parts = []
    for name, mean in means.items():
        values_parts.append(np.full(per_domain, mean, dtype=float))
        domain_parts.extend([name] * per_domain)
    values = np.concatenate(values_parts).reshape(-1, 1)
    labels = np.zeros(values.shape[0], dtype=int)
    domain_labels = np.asarray(domain_parts, dtype=object)
    return values, labels, domain_labels


def test_split_series_domain_holdout_excludes_target_domain():
    values, labels, domain_labels = _three_domain_series(per_domain=4)
    split_spec = DetectionSplitSpec(
        kind=DetectionSplitKind.DOMAIN_HOLDOUT,
        target_domain='B',
    )

    train_values, train_labels, calib_values, calib_labels = _split_series(
        values, labels, split_spec, domain_labels=domain_labels,
    )

    # Домен B полностью уходит в калибровку (unseen), train — это A и C.
    assert train_values.shape[0] == 8
    assert calib_values.shape[0] == 4
    assert np.all(calib_values == 10.0)          # только домен B
    assert not np.any(train_values == 10.0)      # B отсутствует в train
    assert train_labels.shape[0] == train_values.shape[0]
    assert calib_labels.shape[0] == calib_values.shape[0]


def test_split_series_domain_holdout_requires_target_domain():
    values, labels, domain_labels = _three_domain_series()
    split_spec = DetectionSplitSpec(kind=DetectionSplitKind.DOMAIN_HOLDOUT)

    with pytest.raises(ValueError):
        _split_series(values, labels, split_spec, domain_labels=domain_labels)


def test_split_series_domain_holdout_rejects_unknown_target():
    values, labels, domain_labels = _three_domain_series()
    split_spec = DetectionSplitSpec(
        kind=DetectionSplitKind.DOMAIN_HOLDOUT,
        target_domain='Z',
    )

    with pytest.raises(ValueError):
        _split_series(values, labels, split_spec, domain_labels=domain_labels)


def test_split_series_temporal_branch_unaffected_by_domain_labels():
    values, labels, domain_labels = _three_domain_series(per_domain=10)
    split_spec = DetectionSplitSpec(kind=DetectionSplitKind.TEMPORAL)

    train_values, _, calib_values, _ = _split_series(
        values, labels, split_spec, domain_labels=domain_labels,
    )

    # Temporal-ветка игнорирует domain_labels: train идёт раньше calibration.
    assert train_values.shape[0] > 0
    assert calib_values.shape[0] > 0
    assert train_values.shape[0] + calib_values.shape[0] <= values.shape[0]


def test_iter_domain_holdouts_returns_unique_in_order():
    domain_labels = np.asarray(
        ['A', 'A', 'C', 'C', 'B', 'B', 'A'], dtype=object,
    )

    assert iter_domain_holdouts(domain_labels) == ('A', 'C', 'B')


def test_iter_domain_holdouts_raises_with_single_domain():
    domain_labels = np.asarray(['A', 'A', 'A'], dtype=object)

    with pytest.raises(ValueError):
        iter_domain_holdouts(domain_labels)


def test_lodo_loop_runs_on_each_unseen_domain():
    """Тонкая LODO-обёртка поверх run_detection_stage_tuning_on_series.

    По одному прогону на каждый target-домен; метрику усредняем снаружи.
    """
    rng = np.random.default_rng(0)
    per_domain = 120
    means = {'A': 0.0, 'B': 5.0, 'C': 10.0}
    values_parts, domain_parts, label_parts = [], [], []
    for name, mean in means.items():
        block = rng.normal(loc=mean, scale=0.5, size=per_domain)
        block[-5:] += 8.0  # несколько аномалий в хвосте каждого домена
        block_labels = np.zeros(per_domain, dtype=int)
        block_labels[-5:] = 1
        values_parts.append(block)
        domain_parts.extend([name] * per_domain)
        label_parts.append(block_labels)
    values = np.concatenate(values_parts).reshape(-1, 1)
    labels = np.concatenate(label_parts)
    domain_labels = np.asarray(domain_parts, dtype=object)

    per_domain_scores = []
    for target in iter_domain_holdouts(domain_labels):
        result = run_detection_stage_tuning_on_series(
            'feature_iforest_detector',
            values=values,
            labels=labels,
            base_params={'window_size': 16, 'stride': 4},
            metric_name='bin_f1',
            split_spec=DetectionSplitSpec(
                kind=DetectionSplitKind.DOMAIN_HOLDOUT,
                target_domain=target,
            ),
            domain_labels=domain_labels,
            max_values_per_parameter=1,
            max_stage_candidates=1,
            progress_policy=False,
        )
        per_domain_scores.append(result.best_evaluation.metric_value)

    # Прогон отработал на каждом из трёх unseen-доменов и вернул конечные метрики.
    assert len(per_domain_scores) == 3
    assert all(np.isfinite(score) for score in per_domain_scores)
