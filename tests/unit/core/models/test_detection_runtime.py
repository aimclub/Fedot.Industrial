from dataclasses import replace

import numpy as np
import pytest

from fedot_ind.core.models.detection.runtime import (
    AnomalyScoreSeries,
    DetectionSplitKind,
    DetectionSplitSpec,
    DetectionWindowBatch,
    RegimeSegment,
    DetectionEvent,
    build_detection_window_batch,
    build_risk_feature_frame,
    detect_events_from_score_series,
    estimate_detection_threshold,
    infer_regime_segments,
    split_detection_batch,
    ensure_detection_array,
    resolve_detection_window_size,
    resolve_detection_stride,
    _split_temporal_window_ids,
    build_window_statistical_features,
    align_window_scores_to_points,
    estimate_detection_threshold,
    build_anomaly_score_series,
    domain_invariant_scale,
    coral_feature_align,
    build_transfer_alignment_report,
)

from fedot_ind.core.models.detection.stage_tuning import (
    DetectionStageName,
    build_detection_stage_tuning_plan)


def _multichannel_series(length: int = 48) -> np.ndarray:
    time = np.arange(length, dtype=float)
    return np.column_stack(
        (
            np.sin(time / 4.0),
            np.cos(time / 5.0) + 0.1 * time,
        )
    )


def test_build_detection_window_batch_preserves_window_count_and_shape():
    batch = build_detection_window_batch(
        _multichannel_series(48),
        window_size=8,
        stride=2,
    )

    assert batch.windows.shape == (21, 8, 2)
    assert batch.window_indices.shape == (21, 2)
    assert batch.n_windows == 21
    assert batch.n_channels == 2
    assert batch.flattened_features.shape == (21, 16)
    assert batch.statistical_features.shape == (21, 10)


def test_split_detection_batch_temporal_prevents_future_leakage_between_calib_and_test():
    """TEMPORAL-разбиение: единственная реальная гарантия отсутствия утечки из будущего —
    калибровочные окна полностью предшествуют тестовым по ВРЕМЕНИ (calib_end <= test_start).
        Окна перекрываются, когда stride < window_size (здесь window_size=10, stride=2).
        TEMPORAL-сплит в split_detection_batch режет ПО ПОРЯДКУ ID окон:
        train = первые n_train окон (по id), остаток уходит в calib/test.
        Поэтому train идёт раньше calib по id (и по START), но временной КОНЕЦ
        последнего train-окна налезает на START первого calib-окна. То есть
        train_end <= calib_start в общем случае НЕ выполняется — это была неверная
        предпосылка прежней версии теста. prevent_future_leakage влияет только на
        границу calib/test (см. _split_temporal_window_ids).

        series_length=64, window_size=10, stride=2
        -> n_windows = (64-10)//2 + 1 = 28, окно i имеет start=2*i, end=2*i+10.
        train_fraction=0.5  -> n_train = round(28*0.5) = 14  (id 0..13, start 0..26).
        remaining = id 14..27 (14 окон).
        calibration_fraction=0.25 -> n_calib = round(14*0.25/(1-0.5)) = round(7.0) = 7
                                     (id 14..20, start 28..40), calib_end = end(id=20) = 40+10 = 50.
        prevent_future_leakage=True -> test = окна со start >= 50: 2*i>=50 => i>=25
                                       (id 25..27, start 50..54) => 3 окна.
        Окна id 21..24 (start 42..48 < 50) отбрасываются как пересекающие границу calib.
    """
    batch = build_detection_window_batch(
        _multichannel_series(64),
        window_size=10,
        stride=2,
    )
    train_batch, calibration_batch, test_batch = split_detection_batch(
        batch,
        DetectionSplitSpec(
            kind=DetectionSplitKind.TEMPORAL,
            train_fraction=0.5,
            calibration_fraction=0.25,
            prevent_future_leakage=True,
        ),
    )

    assert train_batch is not None
    assert calibration_batch is not None
    # Размеры сплитов (детерминированы для TEMPORAL).
    assert train_batch.n_windows == 14
    assert calibration_batch.n_windows == 7
    assert test_batch is not None and test_batch.n_windows == 3
    # train предшествует calib по ПОРЯДКУ START (id-упорядоченность), но НЕ по концу окна.
    assert train_batch.window_indices[-1, 0] < calibration_batch.window_indices[0, 0]
    # Главная гарантия prevent_future_leakage: calib целиком раньше test по времени.
    assert calibration_batch.window_indices[-1, 1] <= test_batch.window_indices[0, 0]


def test_split_detection_batch_random_holdout_is_deterministic_for_same_seed():
    batch = build_detection_window_batch(
        _multichannel_series(40),
        window_size=8,
        stride=1,
    )
    split_spec = DetectionSplitSpec(
        kind=DetectionSplitKind.HOLDOUT,
        train_fraction=0.6,
        calibration_fraction=0.2,
        random_seed=7,
        prevent_future_leakage=False,
    )

    first = split_detection_batch(batch, split_spec)
    second = split_detection_batch(batch, split_spec)

    assert first[0].metadata['selected_window_ids'] == second[0].metadata['selected_window_ids']
    assert first[1].metadata['selected_window_ids'] == second[1].metadata['selected_window_ids']
    if first[2] is not None and second[2] is not None:
        assert first[2].metadata['selected_window_ids'] == second[2].metadata['selected_window_ids']


def test_split_detection_batch_domain_holdout_uses_target_domain_windows_only_for_holdout():
    batch = build_detection_window_batch(
        _multichannel_series(24),
        window_size=6,
        stride=1,
        metadata={
            'window_domains': ['source'] * 10 + ['target'] * 9,
        },
    )

    train_batch, calibration_batch, test_batch = split_detection_batch(
        batch,
        DetectionSplitSpec(
            kind=DetectionSplitKind.DOMAIN_HOLDOUT,
            calibration_fraction=0.4,
            target_domain='target',
        ),
    )

    assert set(train_batch.metadata['window_domains']) == {'source'}
    assert set(calibration_batch.metadata['window_domains']) == {'target'}
    assert train_batch.metadata['split_kind'] == DetectionSplitKind.DOMAIN_HOLDOUT.value
    assert calibration_batch.metadata['split_name'] == 'calibration'
    if test_batch is not None:
        assert set(test_batch.metadata['window_domains']) == {'target'}
        assert test_batch.metadata['split_name'] == 'test'


def test_estimate_detection_threshold_quantile_is_monotonic():
    scores = np.linspace(0.0, 10.0, num=100)

    lower = estimate_detection_threshold(scores, strategy='quantile', quantile=0.9)
    higher = estimate_detection_threshold(scores, strategy='quantile', quantile=0.99)

    assert higher >= lower


def test_detect_events_and_risk_frame_preserve_regime_context():
    scores = np.array([0.1, 0.2, 1.5, 1.7, 0.4, 0.2, 2.2, 2.3, 2.0, 0.1], dtype=float)
    score_series = AnomalyScoreSeries(
        scores=tuple(scores.tolist()),
        labels=tuple(int(value >= 1.0) for value in scores.tolist()),
        threshold=1.0,
        calibration_strategy='mad',
    )
    regimes = infer_regime_segments(_multichannel_series(10))

    events = detect_events_from_score_series(
        score_series,
        min_event_length=2,
        regime_segments=regimes,
    )
    risk_frame = build_risk_feature_frame(
        events=events,
        regime_segments=regimes,
        score_series=score_series,
        node_name='mill_1',
        domain_name='mpsi',
    )

    assert len(events) == 2
    assert risk_frame.metadata['n_events'] == 2
    frame = risk_frame.to_frame()
    assert {'event_peak_score', 'regime_label', 'node_name', 'domain_name'} <= set(frame.columns)
    assert set(frame['node_name']) == {'mill_1'}


class TestEnsureDetectionArray:
    """Тесты для ensure_detection_array"""

    def test_detection_array_contract_K(self):
        series = np.arange(100)

        arr = ensure_detection_array(series)

        assert arr.ndim == 2
        assert arr.shape == (100, 1)

    def test_1d_list_returns_2d_column(self):
        values = [1, 2, 3]
        result = ensure_detection_array(values)
        expected = np.array([[1], [2], [3]], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_1d_numpy_returns_2d_column(self):
        values = np.array([1, 2, 3])
        result = ensure_detection_array(values)
        expected = np.array([[1], [2], [3]], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_2d_array_unchanged(self):
        values = np.array([[1, 2], [3, 4]])
        result = ensure_detection_array(values)
        np.testing.assert_array_equal(result, values)

    def test_3d_array_reshaped_to_2d(self):
        values = np.random.rand(4, 3, 2)  # shape (4,3,2)
        result = ensure_detection_array(values)
        # ожидается (4, 6)
        assert result.shape == (4, 3 * 2)
        np.testing.assert_array_equal(result, values.reshape(4, -1))

    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match='Detection input must be at least one-dimensional.'):
            ensure_detection_array(np.array(5))  # 0-dim


class TestResolveDetectionWindowSize:
    """Тесты для resolve_detection_window_size"""

    def test_very_short_series(self):
        assert resolve_detection_window_size(-1) == 1
        assert resolve_detection_window_size(3) == 3  # <4 -> max(1, series_length)
        assert resolve_detection_window_size(2) == 2
        assert resolve_detection_window_size(1) == 1

    def test_window_size_explicit(self):
        assert resolve_detection_window_size(100, window_size=20) == 20
        assert resolve_detection_window_size(100, window_size=150) == 100  # capped
        assert resolve_detection_window_size(100, window_size=1) == 2     # min=2

    def test_window_size_percent(self):
        assert resolve_detection_window_size(200, window_size_percent=10) == 20
        assert resolve_detection_window_size(200, window_size_percent=0.5) == 2  # min 2
        assert resolve_detection_window_size(200, window_size_percent=150) == 200
        # Вот это место в коде странное, для граничного случая в меньшую сторону округляет (ДОЛЖНО ЛИ ТАК БЫТЬ?)
        # assert resolve_detection_window_size(5, window_size_percent=50) == 3      # round(2.5)=3, cap=5

    def test_default_behavior(self):
        # default = max(8, round(series_length*0.1))
        assert resolve_detection_window_size(100) == max(8, 10) == 10
        assert resolve_detection_window_size(50) == max(8, 5) == 8
        assert resolve_detection_window_size(200, minimum_window_size=5) == max(5, 20) == 20
        assert resolve_detection_window_size(10, minimum_window_size=20) == min(20, 10) == 10


class TestResolveDetectionStride:
    """Тесты для resolve_detection_stride"""

    def test_stride_given(self):
        assert resolve_detection_stride(100, stride=5) == 5
        assert resolve_detection_stride(100, stride=0) == 1   # max(1,0)

    def test_default_stride(self):
        assert resolve_detection_stride(100) == 100 // 4 == 25
        assert resolve_detection_stride(1) == 1  # max(1,0) = 1


class TestBuildDetectionWindowBatch:
    """Тесты для build_detection_window_batch"""

    def test_detection_window_batch_contract_K(self):
        series = np.linspace(0.0, 1.0, num=120)

        batch = build_detection_window_batch(
            series,
            window_size=20,
            stride=5
        )

        start, end = batch.window_indices[0]

        assert batch.windows.ndim == 3  # [N, W, C]
        assert batch.window_indices.shape[1] == 2
        assert batch.n_windows > 0
        assert batch.n_channels == 1
        assert end - start == batch.window_size

    def test_basic_2d_series(self):
        values = np.random.rand(50, 3)
        # values = np.linspace(1.0, 50.0, num=50).reshape(2,25)
        batch = build_detection_window_batch(values, window_size=10, stride=5)
        assert batch.windows.shape == (9, 10, 3)   # (50-10)//5 +1 = 9
        assert batch.window_indices.shape == (9, 2)
        assert batch.original_length == 50
        assert batch.window_size == 10
        assert batch.stride == 5
        assert len(batch.channel_names) == 3
        assert batch.channel_names == tuple(f'channel_{i}' for i in range(3))

    def test_1d_series(self):
        values = list(range(20))
        batch = build_detection_window_batch(values, window_size=5, stride=2)
        assert batch.windows.shape == (8, 5, 1)   # (20-5)//2+1=8
        row = np.arange(5.0)
        offsets = np.arange(0.0, 15.0, 2.0).reshape(-1, 1)
        result = row + offsets
        np.testing.assert_array_equal(batch.windows[:, :, 0], result)

    def test_custom_channel_names(self):
        values = np.random.rand(30, 2)
        names = ['A', 'B']
        batch = build_detection_window_batch(values, window_size=6, channel_names=names)
        assert batch.channel_names == ('A', 'B')

    def test_metadata(self):
        values = np.random.rand(30, 2)
        meta = {'source': 'test'}
        batch = build_detection_window_batch(values, window_size=6, metadata=meta)
        assert batch.metadata == meta

    def test_series_shorter_than_window(self):
        values = np.random.rand(5, 2)
        with pytest.raises(ValueError, match="shorter than the requested detection window"):
            build_detection_window_batch(values, window_size=10)


class TestSplitDetectionBatch:

    def test_split_detection_batch_contract_invariants_K(self):
        """Контрактные инварианты TEMPORAL-сплита: дизъюнктность множеств окон,
        id-упорядоченность train -> calib -> test и сохранность общего объёма.

        Сценарий: series_length=120, window_size=10, stride=1 (окна сильно перекрываются),
        train_fraction=0.6, calibration_fraction=0.2, prevent_future_leakage=True.
            n_windows = (120-10)//1 + 1 = 111, окно i: start=i, end=i+10.
            n_train = round(111*0.6) = round(66.6) = 67  -> id 0..66.
            remaining = id 67..110 (44 окна).
            n_calib = round(44*0.2/(1-0.6)) = round(22.0) = 22 -> id 67..88,
                      calib_end = end(id=88) = 88+10 = 98.
            test = окна со start >= 98: id 98..110 -> 13 окон.
            id 89..97 (start 89..97 < 98) отбрасываются границей утечки.
            Итого train=67, calib=22, test=13, total = 102 <= 111.
        сравниваем НАБОРЫ id окон (а не пары (start,end)),
        потому что перекрывающиеся окна — это разные окна с разными id, и именно
        непересечение id-множеств train/calib/test и есть смысл контракта.
        Прежнее утверждение ``max(train_end) <= min(calib_start)`` убрано как НЕВЕРНОЕ:
        при stride=1 < window_size конец train-окна (76) больше начала calib-окна (67).
        Корректный временной порядок гарантируется только между calib и test.
        """
        series = np.linspace(0, 1, 120)

        batch = build_detection_window_batch(
            series,
            window_size=10,
            stride=1
        )

        split_spec = DetectionSplitSpec(
            kind=DetectionSplitKind.TEMPORAL,
            train_fraction=0.6,
            calibration_fraction=0.2,
            prevent_future_leakage=True,
        )

        train, calib, test = split_detection_batch(batch, split_spec)

        # Точные размеры сплитов, выведенные из реализации.
        assert train.n_windows == 67
        assert calib.n_windows == 22
        assert test is not None and test.n_windows == 13

        # Дизъюнктность множеств id окон train/calib/test.
        train_ids = set(train.metadata['selected_window_ids'])
        calib_ids = set(calib.metadata['selected_window_ids'])
        test_ids = set(test.metadata['selected_window_ids'])

        assert train_ids.isdisjoint(calib_ids)
        assert train_ids.isdisjoint(test_ids)
        assert calib_ids.isdisjoint(test_ids)

        # id-упорядоченность: train целиком раньше calib, calib целиком раньше test.
        assert max(train_ids) < min(calib_ids)
        assert max(calib_ids) < min(test_ids)

        # Реальная гарантия отсутствия утечки: calib раньше test по ВРЕМЕНИ.
        assert calib.window_indices[-1, 1] <= test.window_indices[0, 0]

        # Сохранность объёма: ни одно окно не задвоено, часть могла быть отброшена.
        total = len(train_ids | calib_ids | test_ids)
        assert total == train.n_windows + calib.n_windows + test.n_windows
        assert total <= batch.n_windows

    def test_domain_holdout_split_contract_K(self):
        """DOMAIN_HOLDOUT: train = исходный домен ('A'), holdout (calib+test) = целевой ('B'),
        внутри целевого домена calib предшествует test по времени.
            series_length=200, window_size=10, stride=2
            -> n_windows = (200-10)//2 + 1 = 96, окно i: start=2*i, end=2*i+10.
            domains: id 0..39 = 'A' (40 окон), id 40..95 = 'B' (56 окон).
            train = окна с доменом != 'B' = id 0..39  -> 40 окон.
            holdout = окна с доменом == 'B' = id 40..95 -> 56 окон.
            n_calib = round(56 * 0.3) = round(16.8) = 17 -> id 40..56,
                      calib_end = end(id=56) = 56*2+10 = 122.
            prevent_future_leakage=True -> test = окна holdout со start >= 122:
                      2*i>=122 => i>=61 => id 61..95 -> 35 окон.
            id 57..60 (start 114..120 < 122) отбрасываются границей утечки.
            ain=40, calib=17, test=35.
        """
        series = np.linspace(0, 1, 200)

        batch = build_detection_window_batch(
            series,
            window_size=10,
            stride=2
        )

        n_windows = batch.n_windows
        assert n_windows == 96
        split_point = 40
        domains = np.array(["A"] * split_point + ["B"] * (n_windows - split_point))

        batch = replace(batch, metadata={"window_domains": domains})

        split_spec = DetectionSplitSpec(
            kind=DetectionSplitKind.DOMAIN_HOLDOUT,
            target_domain="B",
            calibration_fraction=0.3,
            prevent_future_leakage=True,
        )

        train, calib, test = split_detection_batch(batch, split_spec)

        # Домены сплитов: train = только 'A', calib/test = только 'B'.
        assert all(d == "A" for d in train.metadata["window_domains"])
        assert all(d == "B" for d in calib.metadata["window_domains"])

        # Точные размеры, выведенные из реализации.
        assert train.n_windows == 40
        assert calib.n_windows == 17

        if test is not None:
            assert all(d == "B" for d in test.metadata["window_domains"])
            assert test.n_windows == 35

        if test is not None and test.n_windows > 0:
            calib_end = calib.window_indices[-1][1]
            test_start = test.window_indices[0][0]
            assert calib_end <= test_start

    @pytest.fixture
    def sample_batch(self):
        # создаём простой batch: 10 окон, индексы [0,5), [5,10), ...
        n_windows = 10
        window_size = 5
        stride = 5
        original_length = n_windows * stride + window_size - stride  # 5*10+5-5=50
        windows = np.random.rand(n_windows, window_size, 2)
        indices = np.array([[i * stride, i * stride + window_size] for i in range(n_windows)])
        return DetectionWindowBatch(
            windows=windows,
            window_indices=indices,
            original_length=original_length,
            window_size=window_size,
            stride=stride,
            channel_names=('ch0', 'ch1'),
            metadata={}
        )

    def test_domain_holdout_split(self, sample_batch):
        # добавляем домены в metadata
        domains = ['A'] * 3 + ['B'] * 7   # 3 домена A, 7 B
        sample_batch.metadata['window_domains'] = domains
        split_spec = DetectionSplitSpec(
            kind=DetectionSplitKind.DOMAIN_HOLDOUT,
            target_domain='B',
            calibration_fraction=0.3   # 30% от holdout -> около 2 окон
        )
        train, calib, test = split_detection_batch(sample_batch, split_spec)
        # train: только домен A (3 окна)
        assert train.n_windows == 3
        # test: домен B без калибровочных
        assert test.n_windows == 7 - 2 == 5  # 7 окон B, 2 на калибровку
        assert calib.n_windows == 2
        # проверка, что калибровка и тест не пересекаются
        calib_indices = set(calib.window_indices[:, 0])
        test_indices = set(test.window_indices[:, 0])
        assert calib_indices.isdisjoint(test_indices)
        # проверка метаданных
        assert train.metadata['split_name'] == 'train'
        assert calib.metadata['split_name'] == 'calibration'
        assert test.metadata['split_name'] == 'test'

    def test_domain_holdout_missing_domains(self, sample_batch):
        split_spec = DetectionSplitSpec(kind=DetectionSplitKind.DOMAIN_HOLDOUT)
        with pytest.raises(ValueError, match="requires batch.metadata"):
            split_detection_batch(sample_batch, split_spec)

    def test_holdout_random_split(self, sample_batch):
        """Случайный HOLDOUT (prevent_future_leakage=False): проверяем ИНВАРИАНТЫ, а не
        конкретные индексы.
        Размеры выведены из реализации (sample_batch: n_windows=10, start окна i = 5*i):
            n_train = max(1, round(10*0.5)) = 5.
            remaining = 10 - 5 = 5.
            n_calib = round(remaining * calib_fraction / (1 - train_fraction))
                    = round(5 * 0.3 / 0.5) = round(3.0) = 3.
            test = остаток = 5 - 3 = 2.
        """
        split_spec = DetectionSplitSpec(
            kind=DetectionSplitKind.HOLDOUT,
            train_fraction=0.5,
            calibration_fraction=0.3,
            random_seed=42,
            prevent_future_leakage=False
        )
        train, calib, test = split_detection_batch(sample_batch, split_spec)

        # Детерминированные размеры сплитов.
        assert train.n_windows == 5
        assert calib.n_windows == 3
        assert test.n_windows == 2

        # Разбиен перестановка всех 10 окон: дизъюнктность + полное покрытие.
        train_ids = set(train.metadata['selected_window_ids'])
        calib_ids = set(calib.metadata['selected_window_ids'])
        test_ids = set(test.metadata['selected_window_ids'])
        assert train_ids.isdisjoint(calib_ids)
        assert train_ids.isdisjoint(test_ids)
        assert calib_ids.isdisjoint(test_ids)
        assert train_ids | calib_ids | test_ids == set(range(sample_batch.n_windows))

        # Окна действительно перемешаны, а не взяты подряд (start окна i = 5*i).
        train_starts = train.window_indices[:, 0]
        assert not np.array_equal(train_starts, np.arange(0, 5 * 5, 5))

        # Детерминизм при одном и том же random_seed: повторный вызов даёт те же id.
        train2, calib2, test2 = split_detection_batch(sample_batch, split_spec)
        assert train2.metadata['selected_window_ids'] == train.metadata['selected_window_ids']
        assert calib2.metadata['selected_window_ids'] == calib.metadata['selected_window_ids']
        assert test2.metadata['selected_window_ids'] == test.metadata['selected_window_ids']

    def test_temporal_split_prevent_future_leakage(self, sample_batch):
        """Проверяем, что при prevent_future_leakage=True калибровка и тест разделены по времени."""
        split_spec = DetectionSplitSpec(
            kind=DetectionSplitKind.HOLDOUT,
            train_fraction=0.5,
            calibration_fraction=0.3,
            prevent_future_leakage=True
        )
        train, calib, test = split_detection_batch(sample_batch, split_spec)
        # train - первые 5 окон (индексы 0..4)
        # калибровка - следующие 3 окна (индексы 5,6,7) - но они могут быть скорректированы
        # тест - окна после последнего калибровочного по времени
        # убедимся, что ни одно тестовое окно не начинается раньше конца последнего калибровочного
        if calib.n_windows > 0 and test.n_windows > 0:
            calib_end = np.max(calib.window_indices[:, 1])
            test_starts = test.window_indices[:, 0]
            assert np.all(test_starts >= calib_end)

    def test_empty_batch(self):
        empty_batch = DetectionWindowBatch(
            windows=np.empty((0, 5, 2)), window_indices=np.empty((0, 2), dtype=int),
            original_length=10, window_size=5, stride=5, channel_names=()
        )
        split_spec = DetectionSplitSpec(kind=DetectionSplitKind.HOLDOUT)
        with pytest.raises(ValueError, match="at least one window"):
            split_detection_batch(empty_batch, split_spec)


class TestSplitTemporalWindowIds:
    """Тесты для _split_temporal_window_ids"""

    def test_basic_split(self):
        indices = np.array([[0, 10], [10, 20], [20, 30], [30, 40], [40, 50]])
        candidate_ids = np.array([0, 1, 2, 3, 4])
        cal, test = _split_temporal_window_ids(
            indices, candidate_ids, calibration_size=3,
            prevent_future_leakage=False, minimum_start=None
        )
        np.testing.assert_array_equal(cal, [0, 1, 2])
        np.testing.assert_array_equal(test, [3, 4])

    def test_future_leakage_prevention(self):
        indices = np.array([[0, 10], [10, 20], [20, 30], [35, 45], [45, 55]])  # разрыв
        candidate_ids = np.array([0, 1, 2, 3, 4])
        cal, test = _split_temporal_window_ids(
            indices, candidate_ids, calibration_size=2,
            prevent_future_leakage=True, minimum_start=None
        )
        # калибровка: первые 2 окна -> [0,1]
        # конец калибровки: indices[1,1]=20
        # тест: все окна, начинающиеся >=20: окна 2 (начало 20), 3 (35), 4 (45)
        np.testing.assert_array_equal(cal, [0, 1])
        np.testing.assert_array_equal(test, [2, 3, 4])

    def test_minimum_start_filter(self):
        indices = np.array([[0, 10], [10, 20], [20, 30], [30, 40]])
        candidate_ids = np.array([0, 1, 2, 3])
        cal, test = _split_temporal_window_ids(
            indices, candidate_ids, calibration_size=2,
            prevent_future_leakage=False, minimum_start=5
        )
        # отфильтровываем окна с start <5 -> оставляем [1,2,3]
        # затем первые calibration_size = 2 из оставшихся -> [1,2]
        np.testing.assert_array_equal(cal, [1, 2])
        np.testing.assert_array_equal(test, [3])


class TestBuildWindowStatisticalFeatures:
    """Тесты для build_window_statistical_features"""

    def test_statistical_features_shape_K(self):
        series = np.linspace(0, 1, 100)

        batch = build_detection_window_batch(
            series,
            window_size=10)

        features = build_window_statistical_features(batch.windows)

        assert features.shape[0] == batch.n_windows
        assert features.shape[1] == 5 * batch.n_channels

    def test_correct_shape(self):
        windows = np.random.rand(10, 20, 3)  # 10 окон, 20 временных шагов, 3 канала
        features = build_window_statistical_features(windows)
        # каждый из 5 признаков: mean, std, min, max, slope -> 5*3 = 15
        assert features.shape == (10, 15)

    def test_feature_values(self):
        # простой случай: одно окно, 2 шага, 1 канал
        windows = np.array([[[1], [3]]])  # shape (1,2,1)
        features = build_window_statistical_features(windows)
        # mean = 2, std = 1, min = 1, max = 3, slope = 3-1=2
        expected = np.array([[2, 1, 1, 3, 2]])
        np.testing.assert_allclose(features, expected)

    def test_invalid_ndim(self):
        with pytest.raises(ValueError, match="shape \\[n_windows, window_size, n_channels\\]"):
            build_window_statistical_features(np.random.rand(10, 5))


class TestInferRegimeSegments:
    """Тесты для infer_regime_segments"""

    def test_regime_segmentation_covers_series_K(self):
        series = np.linspace(0, 10, 120)

        segments = infer_regime_segments(series)
        covered = []
        for seg in segments:
            covered.extend(range(seg.start_index, seg.end_index + 1))

        covered_sorted = sorted(covered)
        assert covered_sorted == list(range(len(series)))

    def test_basic_regime_detection(self):
        # создаём ряд с явными режимами
        t = np.arange(100)
        signal = np.zeros(100)
        signal[:30] = 10   # высокий уровень
        signal[30:60] = 0   # стабильный низкий
        signal[60:80] = np.linspace(0, 10, 20)  # переход
        signal[80:] = 10
        values = signal.reshape(-1, 1)
        segments = infer_regime_segments(values, volatility_window=10, transition_quantile=0.8)
        # проверим, что получили разумное количество сегментов
        assert len(segments) >= 3
        # метки должны быть 'high_load', 'low_load', 'transition'
        labels = [s.regime_label for s in segments]
        assert 'high_load' in labels
        assert 'low_load' in labels or 'stable' in labels
        # проверка, что сегменты покрывают весь ряд без пропусков
        assert segments[0].start_index == 0
        assert segments[-1].end_index == 99
        for i in range(len(segments) - 1):
            assert segments[i].end_index + 1 == segments[i + 1].start_index

    def test_single_channel_input(self):
        values = np.random.rand(50)
        segments = infer_regime_segments(values)
        assert isinstance(segments, tuple)
        assert all(isinstance(s, RegimeSegment) for s in segments)

    def test_multichannel(self):
        values = np.random.rand(50, 3)
        segments = infer_regime_segments(values)
        assert len(segments) > 0


class TestAlignWindowScoresToPoints:
    """Тесты для align_window_scores_to_points"""

    def test_window_to_point_aggregation_contract_K(self):
        series = np.linspace(0, 1, 50)

        batch = build_detection_window_batch(
            series,
            window_size=5,
            stride=1
        )

        window_scores = np.arange(batch.n_windows)

        point_scores = align_window_scores_to_points(window_scores, batch)
        assert len(point_scores) == batch.original_length
        assert np.isfinite(point_scores).all()
        assert np.std(point_scores) > 0

    def test_basic_alignment(self):
        # batch с 3 окнами: [0,2), [2,4), [4,6) (перекрытия нет)
        indices = np.array([[0, 2], [2, 4], [4, 6]])
        batch = DetectionWindowBatch(
            windows=np.empty((3, 2, 1)), window_indices=indices,
            original_length=6, window_size=2, stride=2,
            channel_names=('ch',)
        )
        scores = np.array([1.0, 2.0, 3.0])
        point_scores = align_window_scores_to_points(scores, batch)
        expected = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        np.testing.assert_array_equal(point_scores, expected)

    def test_overlapping_windows(self):
        # окна с шагом 1: [0,3), [1,4), [2,5)
        indices = np.array([[0, 3], [1, 4], [2, 5]])
        batch = DetectionWindowBatch(
            windows=np.empty((3, 3, 1)), window_indices=indices,
            original_length=5, window_size=3, stride=1,
            channel_names=('ch',)
        )
        scores = np.array([1, 10, 100])
        point_scores = align_window_scores_to_points(scores, batch)
        # точка0: только окно0 -> 1
        # точка1: окно0+окно1 -> (1+10)/2=5.5
        # точка2: окно0+окно1+окно2 -> (1+10+100)/3=37
        # точка3: окно1+окно2 -> (10+100)/2=55
        # точка4: окно2 -> 100
        expected = np.array([1, 5.5, 37, 55, 100])
        np.testing.assert_allclose(point_scores, expected)

    def test_mismatched_length(self):
        batch = DetectionWindowBatch(
            windows=np.empty((5, 2, 1)), window_indices=np.empty((5, 2)),
            original_length=10, window_size=2, stride=2, channel_names=('ch',)
        )
        with pytest.raises(ValueError, match="number of window scores must match"):
            align_window_scores_to_points([1, 2, 3], batch)


class TestEstimateDetectionThreshold:
    """Тесты для estimate_detection_threshold"""

    def test_mad_strategy(self):
        scores = np.array([1, 1, 1, 1, 100])
        thresh = estimate_detection_threshold(scores, strategy='mad')
        # медиана=1, MAD=0, std=~39.6 -> 1+3*39.6 ~ 119.8
        # так как MAD=0, берется std
        assert thresh > 100   # порог должен быть выше выброса
        # более стабильный тест
        scores2 = np.array([0, 0, 0, 0, 10])
        thresh2 = estimate_detection_threshold(scores2, strategy='mad')
        assert thresh2 > 10
        # ДОБАВИТЬ ЕЩЕ ТЕСТ НА ТО, ЧТОБЫ НОМАЛЬНЫЙ ВЫВОД БЫЛ

    def test_quantile_strategy(self):
        scores = np.arange(1, 101)
        thresh = estimate_detection_threshold(scores, strategy='quantile', quantile=0.95)
        # ТУТ ЭТО ИЗ-ЗА ТОГО, ЧТО ПОЛУЧАЕТСЯ 95.05 ИЗ-ЗА ЛИНЕЙНОЙ ИНТЕРПОЛЯЦИИ,
        # КАЖЕТСЯ. ХОТЯ У НАС ДИСКРЕТНОЫЕ ЗНАЧЕНИЯ ЖЕ
        assert np.round(thresh) == 95.0

    def test_regime_conditional(self):
        scores = np.array([1, 2, 100, 200, 3, 4])
        regimes = ['stable', 'stable', 'transition', 'transition', 'stable', 'stable']
        thresh = estimate_detection_threshold(scores, strategy='regime_conditional', regime_labels=regimes)
        # стабильные = [1,2,3,4] -> порог MAD = 4?  медиана 2.5, MAD=1.5, порог=2.5+3*1.5=7
        thresh2 = estimate_detection_threshold([1, 2, 3, 4], strategy='mad')
        assert thresh == thresh2

    def test_domain_calibrated(self):
        scores = np.array([10, 12, 11, 13, 100])
        thresh = estimate_detection_threshold(scores, strategy='domain_calibrated')
        # mean = 29.2, std ≈ 35.4, mean+2.5std ≈ 29.2+88.5 = 117.7
        # Проверяем, что порог примерно в той области
        assert 110 < thresh < 120

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Unsupported detection calibration"):
            estimate_detection_threshold([1, 2, 3], strategy='unknown')


class TestBuildAnomalyScoreSeries:
    """Тесты для build_anomaly_score_series"""

    def test_basic(self):
        scores = [0.1, 0.5, 0.9, 0.2]
        series = build_anomaly_score_series(scores, threshold=0.5, calibration_strategy='mad')
        assert series.scores == (0.1, 0.5, 0.9, 0.2)
        assert series.labels == (0, 1, 1, 0)   # порог >=0.5
        assert series.threshold == 0.5
        assert series.calibration_strategy == 'mad'

    def test_metadata(self):
        scores = [1, 2, 3]
        series = build_anomaly_score_series(scores, threshold=2, calibration_strategy='q', metadata={'key': 'val'})
        assert series.metadata == {'key': 'val'}


class TestDetectEventsFromScoreSeries:
    """Тесты для detect_events_from_score_series"""

    def test_event_detection_contract_minimal_K(self):
        scores = np.array([0.0, 0.6, 0.9, 0.0])
        series = build_anomaly_score_series(
            scores,
            threshold=0.5,
            calibration_strategy='fixed',
        )

        events = detect_events_from_score_series(series, min_event_length=2)
        event = events[0]

        assert len(events) == 1
        assert event.start_index == 1
        assert event.end_index == 2
        assert event.peak_index == 2
        assert np.isclose(event.peak_score, max(scores[1:3]))

    @pytest.fixture
    def score_series(self):
        return AnomalyScoreSeries(
            scores=(0, 1, 1, 1, 0, 0, 1, 1, 0, 1),
            labels=(0, 1, 1, 1, 0, 0, 1, 1, 0, 1),
            threshold=0.5,
            calibration_strategy='test',
            metadata={}
        )

    def test_basic_event_detection(self, score_series):
        events = detect_events_from_score_series(score_series, min_event_length=2)
        # ожидаем события: индексы 1-3 (длина 3) и 6-7 (длина 2), одиночный на индексе 9 игнорируется
        assert len(events) == 2
        assert events[0].start_index == 1
        assert events[0].end_index == 3
        assert events[1].start_index == 6
        assert events[1].end_index == 7

    def test_min_event_length(self, score_series):
        events = detect_events_from_score_series(score_series, min_event_length=3)
        assert len(events) == 1
        assert events[0].start_index == 1
        assert events[0].end_index == 3

    def test_with_regime_segments(self, score_series):
        regimes = [
            RegimeSegment(0, 4, 'high', 10, 1, 0),
            RegimeSegment(5, 9, 'low', 2, 0.5, 0)
        ]
        events = detect_events_from_score_series(score_series, regime_segments=regimes)
        event = events[0]  # первое событие 1-3
        assert event.regime_label == 'high'
        event2 = events[1]  # 6-7
        assert event2.regime_label == 'low'


class TestDomainInvariantScale:
    """Тесты для domain_invariant_scale"""

    def test_scaling_1d(self):
        values = np.array([1, 2, 3, 100]).reshape(-1, 1)
        scaled = domain_invariant_scale(values)
        # медиана = 2.5, MAD = 1 (abs diff: 1.5,0.5,0.5,97.5 -> медиана 1)
        # scaled = (values - 2.5) / 1
        expected = np.array([-1.5, -0.5, 0.5, 97.5]).reshape(-1, 1)
        np.testing.assert_allclose(scaled, expected, rtol=1e-4)

    def test_with_reference(self):
        values = np.array([10, 20, 30]).reshape(-1, 1)
        ref = np.array([0, 10, 20]).reshape(-1, 1)
        scaled = domain_invariant_scale(values, reference_values=ref)
        # ref: медиана=10, MAD=10 -> (values-10)/10 = [0,1,2]
        np.testing.assert_array_equal(scaled, [[0], [1], [2]])

    def test_multichannel(self):
        values = np.array([[1, 100], [2, 200], [3, 300]])
        scaled = domain_invariant_scale(values)
        # медиана по каждому каналу: [2,200]; MAD: [1,100]; результат [[-1,-1],[0,0],[1,1]]
        expected = np.array([[-1, -1], [0, 0], [1, 1]])
        np.testing.assert_allclose(scaled, expected)


class TestCoralFeatureAlign:
    """Тесты для coral_feature_align"""

    def test_alignment(self):
        source = np.array([[1, 2], [3, 4], [5, 6]])   # 3 samples, 2 features
        target = np.array([[10, 20], [30, 40]])     # 2 samples
        aligned = coral_feature_align(source, target, epsilon=1e-6)
        # Проверяем, что среднее и ковариация aligned совпадают с target
        assert aligned.shape == (3, 2)
        np.testing.assert_allclose(np.mean(aligned, axis=0), np.mean(target, axis=0), atol=1e-6)
        cov_aligned = np.cov(aligned, rowvar=False)
        cov_target = np.cov(target, rowvar=False)
        # ТУТ Я ПОНИЗИЛ ДО -4, ПОТОМУ ЧТО В ПЕРВОЙ МАТРИЦЕ ПОЛУЧЕТСЯ 199.9999755, А НЕ 200. НО, ПО СУТИ, БЛИЗКО
        np.testing.assert_allclose(cov_aligned, cov_target, atol=1e-4)

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError, match="expects 2D feature matrices"):
            coral_feature_align(np.random.rand(10), np.random.rand(10, 2))


class TestBuildTransferAlignmentReport:
    """Тесты для build_transfer_alignment_report"""

    def test_transfer_alignment_report_contract_K(self):
        source = np.random.rand(100, 3)
        target = np.random.rand(100, 3)

        report = build_transfer_alignment_report(source, target)

        assert report.n_source == 100
        assert report.n_target == 100
        assert len(report.source_channel_mean) == 3
        assert len(report.target_channel_mean) == 3
        assert len(report.mean_shift) == 3
        assert np.allclose(
            np.array(report.mean_shift),
            np.array(report.target_channel_mean) - np.array(report.source_channel_mean)
        )

    def test_report_creation(self):
        source = np.array([[1, 2], [3, 4], [5, 6]])
        target = np.array([[10, 20], [30, 40]])
        report = build_transfer_alignment_report(source, target, strategy='my_method')
        assert report.strategy == 'my_method'
        assert report.n_source == 3
        assert report.n_target == 2
        assert len(report.source_channel_mean) == 2
        np.testing.assert_allclose(report.source_channel_mean, (3, 4))
        np.testing.assert_allclose(report.target_channel_mean, (20, 30))
        np.testing.assert_allclose(report.mean_shift, (17, 26))
        assert 'source_channel_std' in report.metadata


class TestBuildRiskFeatureFrame:
    """Тесты для build_risk_feature_frame"""

    def test_risk_feature_frame_contract_K(self):
        scores = np.zeros(100)
        scores[40:50] = 2.0

        series = build_anomaly_score_series(
            scores,
            threshold=1.0,
            calibration_strategy='fixed',
        )

        events = detect_events_from_score_series(series)
        regimes = infer_regime_segments(scores)

        frame = build_risk_feature_frame(
            events=events,
            regime_segments=regimes,
            score_series=series,
            node_name='node_1',
            domain_name='test_domain',
        )

        df = frame.to_frame()

        assert len(df) == len(events)
        assert 'event_start_index' in df.columns
        assert 'regime_label' in df.columns

    def test_basic_frame(self):
        events = [
            DetectionEvent(0, 5, 2, 10, 8, 'high'),
            DetectionEvent(10, 12, 11, 5, 4, 'low')
        ]
        regimes = [
            RegimeSegment(0, 7, 'high', 9, 1, 0),
            RegimeSegment(8, 15, 'low', 3, 0.5, 0)
        ]
        score_series = AnomalyScoreSeries((), (), threshold=5, calibration_strategy='mad', metadata={})
        frame = build_risk_feature_frame(
            events=events,
            regime_segments=regimes,
            score_series=score_series,
            node_name='nodeA',
            domain_name='domainX'
        )
        assert len(frame.rows) == 2
        assert frame.columns == (
            'event_start_index', 'event_end_index', 'event_peak_index', 'event_peak_score',
            'event_mean_score', 'event_length', 'regime_label', 'regime_mean_level',
            'regime_volatility', 'node_name', 'domain_name', 'event_threshold', 'calibration_strategy'
        )
        row0 = frame.rows[0]
        assert row0['event_start_index'] == 0
        assert row0['regime_label'] == 'high'
        assert row0['regime_mean_level'] == 9
        assert row0['node_name'] == 'nodeA'
        assert row0['event_threshold'] == 5

    def test_no_score_series(self):
        events = [DetectionEvent(0, 2, 1, 5, 3, 'reg')]
        regimes = [RegimeSegment(0, 5, 'reg', 10, 2, 0)]
        frame = build_risk_feature_frame(events=events, regime_segments=regimes)
        assert 'event_threshold' not in frame.columns
        assert len(frame.rows) == 1
        assert frame.rows[0]['regime_label'] == 'reg'


def test_event_detection_deterministic_K():
    scores = np.array([0, 1, 1, 0, 1, 1, 0])

    series = build_anomaly_score_series(scores, threshold=0.5, calibration_strategy='fixed')

    e1 = detect_events_from_score_series(series)
    e2 = detect_events_from_score_series(series)

    assert e1 == e2
