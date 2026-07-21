import os

import numpy as np
import pytest

from fedot_ind.tools.serialisation.path_lib import EXAMPLES_DATA_PATH
from fedot_ind.core.models.detection.runtime import (
    DetectionSplitKind,
    DetectionSplitSpec,
    build_detection_window_batch,
    build_transfer_alignment_report,
)
from fedot_ind.core.models.detection.stage_tuning_runtime import _split_series

VALVE_FEATURE_COLUMNS = (
    'Accelerometer1RMS',
    'Accelerometer2RMS',
    'Current',
    'Pressure',
    'Temperature',
    'Thermocouple',
    'Voltage',
    'Volume Flow RateRMS',
)
VALVE_N_FEATURES = len(VALVE_FEATURE_COLUMNS)
TEMPERATURE_CHANNEL_INDEX = VALVE_FEATURE_COLUMNS.index('Temperature')

# (window_size=16, stride=4 → 9 окон на домен).
VALVE_CROSS_DOMAIN_ROWS = 50
DETECTION_WINDOW_SIZE = 16
DETECTION_STRIDE = 4
# statistical features: mean, std, min, max, slope × 8 каналов
STAT_FEATURES_PER_WINDOW = 5 * VALVE_N_FEATURES
EXPECTED_WINDOWS_PER_VALVE_SLICE = (
    len(range(0, VALVE_CROSS_DOMAIN_ROWS - DETECTION_WINDOW_SIZE + 1, DETECTION_STRIDE))
)


def _valve_csv_path(valve: str) -> str:
    return os.path.join(
        EXAMPLES_DATA_PATH, 'benchmark', 'detection', 'data', valve, '0.csv',
    )


def _load_valve_slice(valve: str, n_rows: int = VALVE_CROSS_DOMAIN_ROWS):
    """Загружает первые ``n_rows`` строк valve как (values, timestamps, columns)."""
    pd = pytest.importorskip('pandas')
    path = _valve_csv_path(valve)
    if not os.path.exists(path):
        pytest.skip(f'valve dataset is not available: {path}')
    frame = pd.read_csv(path, sep=';', index_col='datetime', nrows=n_rows)
    feature_columns = [
        column for column in frame.columns if column not in {'anomaly', 'changepoint'}
    ]
    values = frame[feature_columns].to_numpy(dtype=float)
    timestamps = [str(value) for value in frame.index.astype(str)]
    return values, timestamps, tuple(feature_columns)


def _load_merged_valve_domains(n_rows: int = VALVE_CROSS_DOMAIN_ROWS):
    """Склеивает valve1 и valve2 в один ряд с domain_labels."""
    valve1_values, _ts1, columns = _load_valve_slice('valve1', n_rows)
    valve2_values, _ts2, _columns2 = _load_valve_slice('valve2', n_rows)
    assert columns == VALVE_FEATURE_COLUMNS

    values = np.vstack([valve1_values, valve2_values])
    labels = np.zeros(values.shape[0], dtype=int)
    domain_labels = np.asarray(
        ['valve1'] * n_rows + ['valve2'] * n_rows,
        dtype=object,
    )
    return values, labels, domain_labels, valve1_values, valve2_values


def _window_statistical_features(values: np.ndarray) -> np.ndarray:
    batch = build_detection_window_batch(
        values,
        window_size=DETECTION_WINDOW_SIZE,
        stride=DETECTION_STRIDE,
        channel_names=VALVE_FEATURE_COLUMNS,
    )
    return batch.statistical_features


def test_split_series_domain_holdout_valve1_train_valve2_unseen():
    """DOMAIN_HOLDOUT: train только на valve1, calibration — unseen valve2.
    Контракт ``_split_series`` из ``stage_tuning_runtime`` для
    ``DetectionSplitKind.DOMAIN_HOLDOUT``: при склеенном ряде
    valve1 + valve2 и ``target_domain='valve2'`` обучающая выборка
    должна содержать **только** valve1, а калибровочная — **только** valve2.

    Первые 50 строк ``examples/.../valve1/0.csv`` и ``valve2/0.csv``
    (8 каналов, шаг ~1 с). Метки аномалий нулевые — здесь важен только
    split по ``domain_labels``, не качество детекции.

    * ``n_rows=50`` на домен → ``train.shape == (50, 8)``, ``calib.shape == (50, 8)``;
    * train/calib **байт-в-байт** совпадают с исходными срезами valve1/valve2;
    * в train нет ни одной строки valve2 (проверка через маску домена и
      канал Temperature: средний уровень train ≈ valve1, calib ≈ valve2);
    * размеры ``labels`` совпадают с ``values``.
    """
    n_rows = VALVE_CROSS_DOMAIN_ROWS
    values, labels, domain_labels, valve1_values, valve2_values = _load_merged_valve_domains(
        n_rows,
    )
    split_spec = DetectionSplitSpec(
        kind=DetectionSplitKind.DOMAIN_HOLDOUT,
        target_domain='valve2',
    )

    train_values, train_labels, calib_values, calib_labels = _split_series(
        values,
        labels,
        split_spec,
        domain_labels=domain_labels,
    )

    assert train_values.shape == (n_rows, VALVE_N_FEATURES)
    assert calib_values.shape == (n_rows, VALVE_N_FEATURES)
    assert train_labels.shape == (n_rows,)
    assert calib_labels.shape == (n_rows,)

    np.testing.assert_array_equal(train_values, valve1_values)
    np.testing.assert_array_equal(calib_values, valve2_values)

    holdout_mask = domain_labels == 'valve2'
    train_mask = ~holdout_mask
    assert holdout_mask.sum() == n_rows
    assert train_mask.sum() == n_rows
    assert not np.any(domain_labels[train_mask] == 'valve2')

    valve1_temp_mean = float(valve1_values[:, TEMPERATURE_CHANNEL_INDEX].mean())
    valve2_temp_mean = float(valve2_values[:, TEMPERATURE_CHANNEL_INDEX].mean())
    assert abs(valve1_temp_mean - valve2_temp_mean) > 1.0

    train_temp_mean = float(train_values[:, TEMPERATURE_CHANNEL_INDEX].mean())
    calib_temp_mean = float(calib_values[:, TEMPERATURE_CHANNEL_INDEX].mean())
    np.testing.assert_allclose(train_temp_mean, valve1_temp_mean, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(calib_temp_mean, valve2_temp_mean, rtol=0.0, atol=1e-9)
    assert abs(train_temp_mean - calib_temp_mean) > 1.0


def test_build_transfer_alignment_report_valve1_to_valve2():
    """Статистический отчёт сдвига доменов valve1 → valve2 на реальных срезах.
    ``build_transfer_alignment_report(source, target)`` на point-level
    значениях двух valve: число наблюдений, 8 каналов в ``mean_shift``,
    формула ``mean_shift = target_mean - source_mean``.

    Те же 50 строк каждого valve CSV (8 каналов без меток).
    * ``n_source == n_target == 50``;
    * длины ``source_channel_mean``, ``target_channel_mean``, ``mean_shift`` == 8;
    * ``mean_shift`` совпадает с разностью средних по каналам;
    * сдвиг Temperature заметно ненулевой (разные operating conditions);
    * в ``metadata`` есть ``source_channel_std`` / ``target_channel_std``.
    """
    _values, _labels, _domains, valve1_values, valve2_values = _load_merged_valve_domains()

    report = build_transfer_alignment_report(
        valve1_values,
        valve2_values,
        strategy='coral',
        source_domain='valve1',
        target_domain='valve2',
    )

    assert report.strategy == 'coral'
    assert report.source_domain == 'valve1'
    assert report.target_domain == 'valve2'
    assert report.n_source == VALVE_CROSS_DOMAIN_ROWS
    assert report.n_target == VALVE_CROSS_DOMAIN_ROWS
    assert len(report.source_channel_mean) == VALVE_N_FEATURES
    assert len(report.target_channel_mean) == VALVE_N_FEATURES
    assert len(report.mean_shift) == VALVE_N_FEATURES

    source_mean = np.mean(valve1_values, axis=0)
    target_mean = np.mean(valve2_values, axis=0)
    expected_shift = target_mean - source_mean

    np.testing.assert_allclose(
        np.array(report.source_channel_mean),
        source_mean,
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.array(report.target_channel_mean),
        target_mean,
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.array(report.mean_shift),
        expected_shift,
        rtol=0.0,
        atol=1e-9,
    )
    assert abs(report.mean_shift[TEMPERATURE_CHANNEL_INDEX]) > 1.0
    assert 'source_channel_std' in report.metadata
    assert 'target_channel_std' in report.metadata
    assert len(report.metadata['source_channel_std']) == VALVE_N_FEATURES
