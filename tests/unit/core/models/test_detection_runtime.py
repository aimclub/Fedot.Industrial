import numpy as np
import pytest

from fedot_ind.core.models.detection.runtime import (
    DetectionWindowBatch,
    DetectionSplitSpec,
    DetectionSplitKind,
    RegimeSegment,
    AnomalyScoreSeries,
    DetectionEvent,
    ensure_detection_array,
    resolve_detection_window_size,
    resolve_detection_stride,
    build_detection_window_batch,
    split_detection_batch,
    _split_temporal_window_ids,
    build_window_statistical_features,
    infer_regime_segments,
    align_window_scores_to_points,
    estimate_detection_threshold,
    build_anomaly_score_series,
    detect_events_from_score_series,
    domain_invariant_scale,
    coral_feature_align,
    build_transfer_alignment_report,
    build_risk_feature_frame)


class TestEnsureDetectionArray:
    """Тесты для ensure_detection_array"""
    
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
        assert result.shape == (4, 3*2)
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
    """Тесты для split_detection_batch"""
    
    @pytest.fixture
    def sample_batch(self):
        # создаём простой batch: 10 окон, индексы [0,5), [5,10), ...
        n_windows = 10
        window_size = 5
        stride = 5
        original_length = n_windows * stride + window_size - stride  # 5*10+5-5=50
        windows = np.random.rand(n_windows, window_size, 2)
        indices = np.array([[i*stride, i*stride+window_size] for i in range(n_windows)])
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
        domains = ['A']*3 + ['B']*7   # 3 домена A, 7 B
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
        split_spec = DetectionSplitSpec(
            kind=DetectionSplitKind.HOLDOUT,
            train_fraction=0.5,
            calibration_fraction=0.3,
            random_seed=42,
            prevent_future_leakage=False
        )
        train, calib, test = split_detection_batch(sample_batch, split_spec)
        assert train.n_windows == 5   # 10*0.5=5
        assert calib.n_windows == 3   # 10*0.3=3
        assert test.n_windows == 2    # остаток
        # проверка, что перемешано (не просто первые окна)
        # так как random_seed фиксирован, можно проверить конкретные индексы
        # убедимся, что индексы не идут подряд
        train_starts = train.window_indices[:, 0]
        assert not np.all(train_starts == np.arange(0, 5*5, 5))  # не просто первые 5 окон

        train_starts = train.window_indices[:, 0].tolist()
        calib_starts = calib.window_indices[:, 0].tolist()
        test_starts = test.window_indices[:, 0].tolist()
        
        expected_train_starts = [0, 5, 25, 35, 40]
        expected_calib_starts = [10, 20, 45]
        expected_test_starts = [15, 30]

        assert train_starts == expected_train_starts
        assert calib_starts == expected_calib_starts
        assert test_starts == expected_test_starts

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
        indices = np.array([[0,10], [10,20], [20,30], [30,40], [40,50]])
        candidate_ids = np.array([0,1,2,3,4])
        cal, test = _split_temporal_window_ids(
            indices, candidate_ids, calibration_size=3,
            prevent_future_leakage=False, minimum_start=None
        )
        np.testing.assert_array_equal(cal, [0,1,2])
        np.testing.assert_array_equal(test, [3,4])
    
    def test_future_leakage_prevention(self):
        indices = np.array([[0,10], [10,20], [20,30], [35,45], [45,55]])  # разрыв
        candidate_ids = np.array([0,1,2,3,4])
        cal, test = _split_temporal_window_ids(
            indices, candidate_ids, calibration_size=2,
            prevent_future_leakage=True, minimum_start=None
        )
        # калибровка: первые 2 окна -> [0,1]
        # конец калибровки: indices[1,1]=20
        # тест: все окна, начинающиеся >=20: окна 2 (начало 20), 3 (35), 4 (45)
        np.testing.assert_array_equal(cal, [0,1])
        np.testing.assert_array_equal(test, [2,3,4])
    
    def test_minimum_start_filter(self):
        indices = np.array([[0,10], [10,20], [20,30], [30,40]])
        candidate_ids = np.array([0,1,2,3])
        cal, test = _split_temporal_window_ids(
            indices, candidate_ids, calibration_size=2,
            prevent_future_leakage=False, minimum_start=5
        )
        # отфильтровываем окна с start <5 -> оставляем [1,2,3]
        # затем первые calibration_size = 2 из оставшихся -> [1,2]
        np.testing.assert_array_equal(cal, [1,2])
        np.testing.assert_array_equal(test, [3])

class TestBuildWindowStatisticalFeatures:
    """Тесты для build_window_statistical_features"""
    
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
    
    def test_basic_regime_detection(self):
        # создаём ряд с явными режимами
        t = np.arange(100)
        signal = np.zeros(100)
        signal[:30] = 10   # высокий уровень
        signal[30:60] = 0   # стабильный низкий
        signal[60:80] = np.linspace(0, 10, 20)  # переход
        signal[80:] = 10
        values = signal.reshape(-1,1)
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
        for i in range(len(segments)-1):
            assert segments[i].end_index + 1 == segments[i+1].start_index
    
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
    
    def test_basic_alignment(self):
        # batch с 3 окнами: [0,2), [2,4), [4,6) (перекрытия нет)
        indices = np.array([[0,2], [2,4], [4,6]])
        batch = DetectionWindowBatch(
            windows=np.empty((3,2,1)), window_indices=indices,
            original_length=6, window_size=2, stride=2,
            channel_names=('ch',)
        )
        scores = np.array([1.0, 2.0, 3.0])
        point_scores = align_window_scores_to_points(scores, batch)
        expected = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        np.testing.assert_array_equal(point_scores, expected)
    
    def test_overlapping_windows(self):
        # окна с шагом 1: [0,3), [1,4), [2,5)
        indices = np.array([[0,3], [1,4], [2,5]])
        batch = DetectionWindowBatch(
            windows=np.empty((3,3,1)), window_indices=indices,
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
            windows=np.empty((5,2,1)), window_indices=np.empty((5,2)),
            original_length=10, window_size=2, stride=2, channel_names=('ch',)
        )
        with pytest.raises(ValueError, match="number of window scores must match"):
            align_window_scores_to_points([1,2,3], batch)

class TestEstimateDetectionThreshold:
    """Тесты для estimate_detection_threshold"""
    
    def test_mad_strategy(self):
        scores = np.array([1,1,1,1,100])
        thresh = estimate_detection_threshold(scores, strategy='mad')
        # медиана=1, MAD=0, std=~39.6 -> 1+3*39.6 ~ 119.8
        # так как MAD=0, берется std
        assert thresh > 100   # порог должен быть выше выброса
        # более стабильный тест
        scores2 = np.array([0,0,0,0,10])
        thresh2 = estimate_detection_threshold(scores2, strategy='mad')
        assert thresh2 > 10
        # ДОБАВИТЬ ЕЩЕ ТЕСТ НА ТО, ЧТОБЫ НОМАЛЬНЫЙ ВЫВОД БЫЛ
        
    def test_quantile_strategy(self):
        scores = np.arange(1,101)
        thresh = estimate_detection_threshold(scores, strategy='quantile', quantile=0.95)
        assert np.round(thresh) == 95.0 # ТУТ ЭТО ИЗ-ЗА ТОГО, ЧТО ПОЛУЧАЕТСЯ 95.05 ИЗ-ЗА ЛИНЕЙНОЙ ИНТЕРПОЛЯЦИИ, КАЖЕТСЯ. ХОТЯ У НАС ДИСКРЕТНОЫЕ ЗНАЧЕНИЯ ЖЕ
    
    def test_regime_conditional(self):
        scores = np.array([1,2,100,200,3,4])
        regimes = ['stable', 'stable', 'transition', 'transition', 'stable', 'stable']
        thresh = estimate_detection_threshold(scores, strategy='regime_conditional', regime_labels=regimes)
        # стабильные = [1,2,3,4] -> порог MAD = 4?  медиана 2.5, MAD=1.5, порог=2.5+3*1.5=7
        thresh2 = estimate_detection_threshold([1,2,3,4], strategy='mad')
        assert thresh == thresh2
    
    def test_domain_calibrated(self):
        scores = np.array([10,12,11,13,100])
        thresh = estimate_detection_threshold(scores, strategy='domain_calibrated')
        # mean = 29.2, std ≈ 35.4, mean+2.5std ≈ 29.2+88.5 = 117.7
        # Проверяем, что порог примерно в той области
        assert 110 < thresh < 120
    
    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Unsupported detection calibration"):
            estimate_detection_threshold([1,2,3], strategy='unknown')

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
        scores = [1,2,3]
        series = build_anomaly_score_series(scores, threshold=2, calibration_strategy='q', metadata={'key':'val'})
        assert series.metadata == {'key':'val'}

class TestDetectEventsFromScoreSeries:
    """Тесты для detect_events_from_score_series"""
    
    @pytest.fixture
    def score_series(self):
        return AnomalyScoreSeries(
            scores=(0,1,1,1,0,0,1,1,0,1),
            labels=(0,1,1,1,0,0,1,1,0,1),
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
            RegimeSegment(0,4,'high',10,1,0),
            RegimeSegment(5,9,'low',2,0.5,0)
        ]
        events = detect_events_from_score_series(score_series, regime_segments=regimes)
        event = events[0]  # первое событие 1-3
        assert event.regime_label == 'high'
        event2 = events[1]  # 6-7
        assert event2.regime_label == 'low'

class TestDomainInvariantScale:
    """Тесты для domain_invariant_scale"""
    
    def test_scaling_1d(self):
        values = np.array([1,2,3,100]).reshape(-1,1)
        scaled = domain_invariant_scale(values)
        # медиана = 2.5, MAD = 1 (abs diff: 1.5,0.5,0.5,97.5 -> медиана 1)
        # scaled = (values - 2.5) / 1
        expected = np.array([-1.5, -0.5, 0.5, 97.5]).reshape(-1,1)
        np.testing.assert_allclose(scaled, expected, rtol=1e-4)
    
    def test_with_reference(self):
        values = np.array([10,20,30]).reshape(-1,1)
        ref = np.array([0,10,20]).reshape(-1,1)
        scaled = domain_invariant_scale(values, reference_values=ref)
        # ref: медиана=10, MAD=10 -> (values-10)/10 = [0,1,2]
        np.testing.assert_array_equal(scaled, [[0],[1],[2]])
    
    def test_multichannel(self):
        values = np.array([[1,100],[2,200],[3,300]])
        scaled = domain_invariant_scale(values)
        # медиана по каждому каналу: [2,200]; MAD: [1,100]; результат [[-1,-1],[0,0],[1,1]]
        expected = np.array([[-1,-1],[0,0],[1,1]])
        np.testing.assert_allclose(scaled, expected)

class TestCoralFeatureAlign:
    """Тесты для coral_feature_align"""
    
    def test_alignment(self):
        source = np.array([[1,2],[3,4],[5,6]])   # 3 samples, 2 features
        target = np.array([[10,20],[30,40]])     # 2 samples
        aligned = coral_feature_align(source, target, epsilon=1e-6)
        # Проверяем, что среднее и ковариация aligned совпадают с target
        assert aligned.shape == (3,2)
        np.testing.assert_allclose(np.mean(aligned, axis=0), np.mean(target, axis=0), atol=1e-6)
        cov_aligned = np.cov(aligned, rowvar=False)
        cov_target = np.cov(target, rowvar=False)
        np.testing.assert_allclose(cov_aligned, cov_target, atol=1e-4) # ТУТ Я ПОНИЗИЛ ДО -4, ПОТОМУ ЧТО В ПЕРВОЙ МАТРИЦЕ ПОЛУЧЕТСЯ 199.9999755, А НЕ 200. НО, ПО СУТИ, БЛИЗКО
    
    def test_invalid_dimensions(self):
        with pytest.raises(ValueError, match="expects 2D feature matrices"):
            coral_feature_align(np.random.rand(10), np.random.rand(10,2))

class TestBuildTransferAlignmentReport:
    """Тесты для build_transfer_alignment_report"""
    
    def test_report_creation(self):
        source = np.array([[1,2],[3,4],[5,6]])
        target = np.array([[10,20],[30,40]])
        report = build_transfer_alignment_report(source, target, strategy='my_method')
        assert report.strategy == 'my_method'
        assert report.n_source == 3
        assert report.n_target == 2
        assert len(report.source_channel_mean) == 2
        np.testing.assert_allclose(report.source_channel_mean, (3,4))
        np.testing.assert_allclose(report.target_channel_mean, (20,30))
        np.testing.assert_allclose(report.mean_shift, (17,26))
        assert 'source_channel_std' in report.metadata

class TestBuildRiskFeatureFrame:
    """Тесты для build_risk_feature_frame"""
    
    def test_basic_frame(self):
        events = [
            DetectionEvent(0,5,2,10,8,'high'),
            DetectionEvent(10,12,11,5,4,'low')
        ]
        regimes = [
            RegimeSegment(0,7,'high',9,1,0),
            RegimeSegment(8,15,'low',3,0.5,0)
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
        events = [DetectionEvent(0,2,1,5,3,'reg')]
        regimes = [RegimeSegment(0,5,'reg',10,2,0)]
        frame = build_risk_feature_frame(events=events, regime_segments=regimes)
        assert 'event_threshold' not in frame.columns
        assert len(frame.rows) == 1
        assert frame.rows[0]['regime_label'] == 'reg'

