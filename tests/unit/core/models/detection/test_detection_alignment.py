import os

import numpy as np
import pytest

from fedot_ind.tools.serialisation.path_lib import EXAMPLES_DATA_PATH
from fedot_ind.core.models.detection.data_quality import (
    align_timestamps,
    prepare_detection_series,
    _timestamps_to_seconds,
    _resolve_duplicates,
    _resample_to_grid,
    _forward_fill,
    _interpolate_linear,
)

# Ожидаемые имена фич-колонок valve-файлов (без меток anomaly/changepoint).
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
VALVE_SLICE_ROWS = 20


def _valve_csv_path(valve: str) -> str:
    """Абсолютный путь к 0.csv"""
    return os.path.join(
        EXAMPLES_DATA_PATH, 'benchmark', 'detection', 'data', valve, '0.csv',
    )


def _load_valve_slice(valve: str, n_rows: int = VALVE_SLICE_ROWS):
    """Загружает первые ``n_rows`` строк valve как (values, timestamps, columns).

    * values     — np.ndarray формы (n_rows, 8) только по фич-колонкам;
    * timestamps — список строк datetime (как они придут в align_timestamps);
    * columns    — кортеж имён фич-колонок.

    Если csv отсутствует (например, данные не выкачаны в окружении) — тест
    пропускается через pytest.skip, чтобы не падать на CI без данных.
    """
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


# Сортировка по времени

def test_align_sorts_values_by_time_when_order_is_broken():
    """Нарушенный порядок timestamps должен восстанавливаться стабильной сортировкой.

    Вход: значения [[10],[20],[30]] с метками времени [2.0, 0.0, 1.0].
    align_timestamps сортирует по возрастанию времени (np.argsort, kind='stable'),
    поэтому:
        * aligned_timestamps == [0.0, 1.0, 2.0];
        * aligned_values     == [[20],[30],[10]] (строки переставлены вслед за временем);
        * gap_mask           == [False, False, False] (равномерный шаг, без NaN);
        * report['n_duplicates'] == 0, report['has_timestamps'] is True;
        * nominal_dt = median положительных разностей == 1.0;
        * resampling_method == 'none' (target_sample_rate_hz не задан).
    """
    values = np.array([[10.0], [20.0], [30.0]])  # TODO: проверить на формат как 2020-03-09 10:14:33
    timestamps = [2.0, 0.0, 1.0]

    aligned_values, aligned_ts, gap_mask, report = align_timestamps(values, timestamps)

    np.testing.assert_array_equal(aligned_ts, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(aligned_values, np.array([[20.0], [30.0], [10.0]]))
    np.testing.assert_array_equal(gap_mask, np.array([False, False, False]))
    assert report['n_duplicates'] == 0
    assert report['has_timestamps'] is True
    assert report['nominal_dt_seconds'] == 1.0
    assert report['resampling_method'] == 'none'


# 2. Дедупликация по политикам keep_last / mean / drop

def test_resolve_duplicates_keep_last_keeps_latest_row():
    """keep_last: для повторяющегося времени остаётся ПОСЛЕДНЕЕ вхождение.

    Вход: seconds=[0,0,1], values=[[1],[2],[3]].
    Для t=0 встречаются строки [1] и [2]; keep_last берёт последнюю → [2].
    Ожидается:
        * series == [[2],[3]];
        * seconds == [0,1];
        * n_duplicates == 1 (число удалённых дублей = len - n_unique = 3 - 2).
    """
    series = np.array([[1.0], [2.0], [3.0]])
    seconds = np.array([0.0, 0.0, 1.0])

    out_series, out_seconds, n_duplicates = _resolve_duplicates(series, seconds, 'keep_last')

    np.testing.assert_array_equal(out_series, np.array([[2.0], [3.0]]))
    np.testing.assert_array_equal(out_seconds, np.array([0.0, 1.0]))
    assert n_duplicates == 1


def test_resolve_duplicates_mean_averages_collisions():
    """mean: для повторяющегося времени значения усредняются поканально.

    Вход: seconds=[0,0,1], values=[[1],[3],[10]].
    Для t=0: (1+3)/2 = 2.0; для t=1: 10.0.
    Ожидается:
        * series == [[2.0],[10.0]];
        * seconds == [0,1];
        * n_duplicates == 1.
    """
    series = np.array([[1.0], [3.0], [10.0]])
    seconds = np.array([0.0, 0.0, 1.0])

    out_series, out_seconds, n_duplicates = _resolve_duplicates(series, seconds, 'mean')

    np.testing.assert_allclose(out_series, np.array([[2.0], [10.0]]))
    np.testing.assert_array_equal(out_seconds, np.array([0.0, 1.0]))
    assert n_duplicates == 1


def test_resolve_duplicates_drop_removes_all_collided_rows():
    """drop: ВСЕ строки с конфликтующим временем выбрасываются целиком.

    Вход: seconds=[0,0,1,2], values=[[1],[3],[10],[20]].
    t=0 встречается дважды → обе строки удаляются; уникальные t=1,2 остаются.
    Особенность контракта: для 'drop' возвращается число УДАЛЁННЫХ строк
    (np.count_nonzero(~keep)), т.е. 2, а не len-n_unique.
    Ожидается:
        * series == [[10],[20]];
        * seconds == [1,2];
        * n_duplicates == 2.
    """
    series = np.array([[1.0], [3.0], [10.0], [20.0]])
    seconds = np.array([0.0, 0.0, 1.0, 2.0])

    out_series, out_seconds, n_duplicates = _resolve_duplicates(series, seconds, 'drop')

    np.testing.assert_array_equal(out_series, np.array([[10.0], [20.0]]))
    np.testing.assert_array_equal(out_seconds, np.array([1.0, 2.0]))
    assert n_duplicates == 2


def test_align_timestamps_reports_duplicates_for_each_policy():
    """Сквозная проверка report['n_duplicates'] через align_timestamps по политикам.

    Один и тот же вход (метки [0,0,1], значения [[1],[3],[5]]) даёт:
        * keep_last → n_duplicates == 1, остаётся [[3],[5]];
        * mean      → n_duplicates == 1, остаётся [[2],[5]] (среднее (1+3)/2);
        * drop      → n_duplicates == 1 (удалена 1 конфликтующая строка t=0,
          но т.к. удаляются обе строки t=0, остаётся только [[5]]).
    Для drop удаляются ОБЕ строки t=0, поэтому остаётся [[5]] и seconds=[1].
    """
    values = np.array([[1.0], [3.0], [5.0]])
    timestamps = [0.0, 0.0, 1.0]

    keep_vals, _ts, _mask, keep_report = align_timestamps(values, timestamps, duplicate_policy='keep_last')
    np.testing.assert_array_equal(keep_vals, np.array([[3.0], [5.0]]))
    assert keep_report['n_duplicates'] == 1

    mean_vals, _ts, _mask, mean_report = align_timestamps(values, timestamps, duplicate_policy='mean')
    np.testing.assert_allclose(mean_vals, np.array([[2.0], [5.0]]))
    assert mean_report['n_duplicates'] == 1

    drop_vals, _ts, _mask, drop_report = align_timestamps(values, timestamps, duplicate_policy='drop')
    np.testing.assert_array_equal(drop_vals, np.array([[5.0]]))
    assert drop_report['n_duplicates'] == 2


# 3. Обнаружение пропусков (n_gaps, gap_total_seconds)

def test_align_detects_time_gaps_without_resampling():
    """Пропуск фиксируется, когда интервал > 1.5 * nominal_dt (без ресемплинга).

    Вход: seconds=[0,1,2,5], values=[[0],[1],[2],[3]], target_sample_rate_hz не задан.
    diffs = [1,1,3]; nominal_dt = median положительных diffs = 1.0.
    gap_factor = diffs/nominal_dt = [1,1,3]; порог 1.5 → разрыв только на 3.0.
    Ожидается:
        * report['n_gaps'] == 1;
        * report['gap_total_seconds'] == 3 - 1 = 2.0 (избыточное время сверх nominal_dt);
        * resampling_method == 'none' (без target_sample_rate_hz сетка не строится);
        * aligned_n == original_n == 4 (ряд только отсортирован, точки не добавлялись).
    """
    values = np.array([[0.0], [1.0], [2.0], [3.0]])
    timestamps = [0.0, 1.0, 2.0, 5.0]

    aligned_values, _ts, _mask, report = align_timestamps(values, timestamps)

    assert report['n_gaps'] == 1
    assert report['gap_total_seconds'] == pytest.approx(2.0)
    assert report['resampling_method'] == 'none'
    assert report['original_n'] == 4
    assert report['aligned_n'] == 4
    assert aligned_values.shape == (4, 1)


# 4. Ресемплинг на регулярную сетку при target_sample_rate_hz

def test_align_resamples_to_regular_grid_mark_only():
    """target_sample_rate_hz=1 строит сетку 1 Гц; пропущенная точка остаётся NaN.

    Вход: seconds=[0,1,2,4] (нет t=3), values=[[10],[20],[30],[50]],
    target_sample_rate_hz=1.0, gap_policy='mark_only' (по умолчанию).
    nominal_dt = 1/1 = 1.0; сетка = arange(0, 4 + 0.5, 1) = [0,1,2,3,4] (5 точек).
    position наблюдений = [0,1,2,4]; точка t=3 (индекс 3) синтетическая.
    Ожидается:
        * aligned_timestamps == [0,1,2,3,4];
        * aligned_values     == [[10],[20],[30],[nan],[50]] (mark_only НЕ заполняет);
        * gap_mask           == [F,F,F,T,F] (только синтетическая/NaN-точка недостоверна);
        * report['aligned_n'] == 5, resampling_method == 'mark_only';
        * report['n_gaps'] == 1 (считается ПО исходным diffs до сетки:
          diffs=[1,1,2], gap_factor=[1,1,2] → один разрыв на 2.0);
        * report['gap_total_seconds'] == 2 - 1 = 1.0.
    Также важна согласованность длины: len(gap_mask) == len(aligned_values) == 5.
    """
    values = np.array([[10.0], [20.0], [30.0], [50.0]])
    timestamps = [0.0, 1.0, 2.0, 4.0]

    aligned_values, aligned_ts, gap_mask, report = align_timestamps(
        values, timestamps, target_sample_rate_hz=1.0,
    )

    np.testing.assert_array_equal(aligned_ts, np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(gap_mask, np.array([False, False, False, True, False]))
    assert np.isnan(aligned_values[3, 0])
    np.testing.assert_array_equal(aligned_values[[0, 1, 2, 4], 0], np.array([10.0, 20.0, 30.0, 50.0]))
    assert report['aligned_n'] == 5
    assert report['resampling_method'] == 'mark_only'
    assert report['n_gaps'] == 1
    assert report['gap_total_seconds'] == pytest.approx(1.0)
    assert len(gap_mask) == aligned_values.shape[0] == 5


def test_resample_to_grid_positions_and_gap_mask_directly():
    """Прямой тест _resample_to_grid: позиции наблюдений и базовый gap_mask.

    Вход: series=[[1],[2],[3]], seconds=[0,2,4], nominal_dt=2.0, gap_policy='mark_only'.
    grid = arange(0, 4 + 1.0, 2) = [0,2,4] (3 точки) — здесь сетка совпадает с
    наблюдениями, синтетических точек нет.
    Возьмём более показательный случай ниже отдельным под-блоком: seconds=[0,2],
    nominal_dt=1.0 → grid=[0,1,2], наблюдения в позициях 0 и 2, точка 1 — синтетическая.
    Ожидается для второго случая:
        * grid == [0,1,2];
        * aligned == [[1],[nan],[2]];
        * gap_mask == [F,T,F];
        * method == 'mark_only'.
    """
    series = np.array([[1.0], [2.0]])
    seconds = np.array([0.0, 2.0])

    aligned, grid, gap_mask, method = _resample_to_grid(series, seconds, 1.0, 'mark_only')

    np.testing.assert_array_equal(grid, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(gap_mask, np.array([False, True, False]))
    assert np.isnan(aligned[1, 0])
    np.testing.assert_array_equal(aligned[[0, 2], 0], np.array([1.0, 2.0]))
    assert method == 'mark_only'


# ---------------------------------------------------------------------------
# 5. Заполнение пропусков: forward_fill и interpolate_linear
# ---------------------------------------------------------------------------

def test_forward_fill_locf_and_backfill_leading_nans():
    """_forward_fill: LOCF (последнее наблюдение вперёд) + backfill ведущих NaN.

    Одноканальный вход [nan, nan, 5, nan, 7, nan]:
        * LOCF протягивает 5 на индекс 3 и 7 на индекс 5;
        * ведущие NaN (индексы 0,1) заполняются первым валидным значением 5.
    Ожидается: [5, 5, 5, 5, 7, 7].

    Многоканальный вход проверяет независимость каналов:
        канал0 [nan,1,nan,4] → [1,1,1,4] (ведущий NaN backfill=1, LOCF 1→2);
        канал1 [2,nan,nan,nan] → [2,2,2,2] (LOCF от первого значения 2).
    Ожидается: [[1,2],[1,2],[1,2],[4,2]].
    """
    single = np.array([[np.nan], [np.nan], [5.0], [np.nan], [7.0], [np.nan]])
    filled_single = _forward_fill(single)
    np.testing.assert_array_equal(
        filled_single[:, 0], np.array([5.0, 5.0, 5.0, 5.0, 7.0, 7.0]),
    )

    multi = np.array([
        [np.nan, 2.0],
        [1.0, np.nan],
        [np.nan, np.nan],
        [4.0, np.nan],
    ])
    filled_multi = _forward_fill(multi)
    np.testing.assert_array_equal(
        filled_multi,
        np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [4.0, 2.0]]),
    )


def test_interpolate_linear_exact_values():
    """_interpolate_linear: линейная интерполяция между валидными точками.

    Вход [nan, nan, 2, nan, 6, nan]; валидные индексы = [2, 4] со значениями [2, 6].
    np.interp(arange(6), [2,4], [2,6]):
        * x<2 (индексы 0,1) клиппируется к крайнему значению 2;
        * x=3 → середина между (2,2) и (4,6) → 4;
        * x>4 (индекс 5) клиппируется к 6.
    Ожидается: [2, 2, 2, 4, 6, 6].
    """
    column = np.array([[np.nan], [np.nan], [2.0], [np.nan], [6.0], [np.nan]])
    filled = _interpolate_linear(column)
    np.testing.assert_array_equal(
        filled[:, 0], np.array([2.0, 2.0, 2.0, 4.0, 6.0, 6.0]),
    )


def test_align_interpolate_linear_fills_resampled_gap():
    """Сквозной gap_policy='interpolate_linear' на ресемплированной сетке.

    Вход: seconds=[0,1,3], values=[[0],[10],[30]], target_sample_rate_hz=1.0.
    nominal_dt=1.0; сетка = arange(0, 3 + 0.5, 1) = [0,1,2,3]; точка t=2 синтетическая.
    До заполнения колонка = [0,10,nan,30]; интерполяция даёт значение 20 на индексе 2.
    Ключевая проверка: gap_mask считается ДО заполнения, поэтому индекс 2 остаётся
    помеченным как недостоверный (True), хотя значение уже заполнено числом 20.
    Ожидается:
        * aligned_values == [[0],[10],[20],[30]];
        * gap_mask == [F,F,T,F];
        * resampling_method == 'interpolate_linear';
        * len(gap_mask) == len(aligned_values) == 4.
    """
    values = np.array([[0.0], [10.0], [30.0]])
    timestamps = [0.0, 1.0, 3.0]

    aligned_values, _ts, gap_mask, report = align_timestamps(
        values, timestamps, target_sample_rate_hz=1.0, gap_policy='interpolate_linear',
    )

    np.testing.assert_allclose(aligned_values[:, 0], np.array([0.0, 10.0, 20.0, 30.0]))
    np.testing.assert_array_equal(gap_mask, np.array([False, False, True, False]))
    assert report['resampling_method'] == 'interpolate_linear'
    assert len(gap_mask) == aligned_values.shape[0] == 4


# 6. Поведение без timestamps (timestamps=None)

def test_align_without_timestamps_uses_integer_index_and_nan_mask():
    """timestamps=None → регулярный целочисленный индекс, gap_mask только по NaN.

    Вход: values=[[1],[2],[nan],[4]], timestamps=None.
    Ожидается:
        * aligned_values возвращается без изменений (включая NaN на индексе 2);
        * aligned_timestamps == arange(4) == [0,1,2,3] (фиктивная регулярная ось);
        * gap_mask == [F,F,T,F] (True только там, где NaN хотя бы в одном канале);
        * report['has_timestamps'] is False;
        * report['nominal_dt_seconds'] == 0.0, resampling_method == 'none';
        * report['n_gaps'] == 0, n_duplicates == 0, aligned_n == 4.
    """
    values = np.array([[1.0], [2.0], [np.nan], [4.0]])

    aligned_values, aligned_ts, gap_mask, report = align_timestamps(values, None)

    np.testing.assert_array_equal(aligned_ts, np.array([0.0, 1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(gap_mask, np.array([False, False, True, False]))
    assert np.isnan(aligned_values[2, 0])
    assert report['has_timestamps'] is False
    assert report['nominal_dt_seconds'] == 0.0
    assert report['resampling_method'] == 'none'
    assert report['n_gaps'] == 0
    assert report['n_duplicates'] == 0
    assert report['aligned_n'] == 4


# 7. Согласованность gap_mask (длина и порядок относительно заполнения)

def test_gap_mask_length_matches_series_and_marks_before_fill():
    """gap_mask всегда длиной с выровненный ряд и помечает синтетику ДО заполнения.

    Сравниваем две политики на одном входе seconds=[0,2], values=[[10],[30]],
    target_sample_rate_hz=1.0 (синтетическая точка на t=1):
        * forward_fill: значение индекса 1 = 10 (LOCF), но gap_mask[1] == True;
        * mark_only:    значение индекса 1 = NaN,        gap_mask[1] == True.
    В обоих случаях:
        * len(gap_mask) == aligned_values.shape[0] == 3;
        * gap_mask == [F,T,F].
    Это подтверждает: маска отражает достоверность исходного измерения, а не факт
    наличия числа после заполнения.
    """
    values = np.array([[10.0], [30.0]])
    timestamps = [0.0, 2.0]

    ff_values, _ts, ff_mask, _report = align_timestamps(
        values, timestamps, target_sample_rate_hz=1.0, gap_policy='forward_fill',
    )
    assert len(ff_mask) == ff_values.shape[0] == 3
    np.testing.assert_array_equal(ff_mask, np.array([False, True, False]))
    assert ff_values[1, 0] == 10.0  # LOCF заполнил, но точка всё равно помечена

    mo_values, _ts, mo_mask, _report = align_timestamps(
        values, timestamps, target_sample_rate_hz=1.0, gap_policy='mark_only',
    )
    assert len(mo_mask) == mo_values.shape[0] == 3
    np.testing.assert_array_equal(mo_mask, np.array([False, True, False]))
    assert np.isnan(mo_values[1, 0])


# 8. Сохранение числа каналов (синтетический + valve-срез)

def test_align_preserves_channel_count_synthetic():
    """Многоканальный вход сохраняет число каналов при сортировке.

    Вход формы (5, 3) с перемешанными timestamps [4,3,2,1,0]; без ресемплинга.
    Ожидается:
        * aligned_values.shape == (5, 3) — число каналов не меняется;
        * строки переставлены в порядке возрастания времени, т.е. реверс входа.
    """
    rng = np.random.default_rng(0)
    values = rng.normal(size=(5, 3))
    timestamps = [4.0, 3.0, 2.0, 1.0, 0.0]

    aligned_values, aligned_ts, _mask, report = align_timestamps(values, timestamps)

    assert aligned_values.shape == (5, 3)
    np.testing.assert_array_equal(aligned_ts, np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(aligned_values, values[::-1])
    assert report['aligned_n'] == 5


@pytest.mark.parametrize('valve', ['valve1', 'valve2'])
def test_valve_slice_shape_is_preserved(valve):
    """Для среза valve (20 строк, 8 фич) форма выровненного ряда == (20, 8).

    Без target_sample_rate_hz ряд только сортируется/дедуплицируется (дубликатов
    времени в первых 20 строках нет), поэтому число строк остаётся 20, а число
    каналов равно числу фич-колонок (8). Это контрактная проверка размерностей
    на реальных данных без зависимости от конкретных числовых значений.
    """
    values, timestamps, columns = _load_valve_slice(valve)
    assert values.shape == (VALVE_SLICE_ROWS, VALVE_N_FEATURES)
    assert columns == VALVE_FEATURE_COLUMNS

    aligned_values, _ts, gap_mask, report = align_timestamps(values, timestamps)

    assert aligned_values.shape == (VALVE_SLICE_ROWS, VALVE_N_FEATURES)
    assert report['original_n'] == VALVE_SLICE_ROWS
    assert report['aligned_n'] == VALVE_SLICE_ROWS
    assert len(gap_mask) == VALVE_SLICE_ROWS


# 9. Парсинг datetime-строк valve в секунды и nominal_dt

def test_timestamps_to_seconds_parses_iso_datetime_strings():
    """_timestamps_to_seconds корректно парсит ISO-строки valve в секунды.

    Берём три подряд идущие метки valve-формата с шагом 1 c:
        '2020-03-09 10:14:33', ':34', ':35'.
    pd.to_datetime → int64 наносекунды → / 1e9 секунд. Проверяем НЕ абсолютные
    эпохи (зависят от таймзоны интерпретации), а разности соседних значений:
        np.diff(seconds) == [1.0, 1.0].
    """
    timestamps = ['2020-03-09 10:14:33', '2020-03-09 10:14:34', '2020-03-09 10:14:35']
    seconds = _timestamps_to_seconds(timestamps)
    assert seconds.shape == (3,)
    np.testing.assert_allclose(np.diff(seconds), np.array([1.0, 1.0]))


def test_timestamps_to_seconds_handles_numeric_strings():
    """Числовые строки ('0.0','1.5','3.0') приводятся к float напрямую без datetime.

    Контракт: если массив можно сконвертировать в float, datetime-парсинг не
    вызывается. Ожидается seconds == [0.0, 1.5, 3.0].
    """
    seconds = _timestamps_to_seconds(['0.0', '1.5', '3.0'])
    np.testing.assert_allclose(seconds, np.array([0.0, 1.5, 3.0]))


@pytest.mark.parametrize('valve', ['valve1', 'valve2'])
def test_valve_slice_nominal_dt_is_one_second(valve):
    """Срез valve имеет шаг дискретизации 1 c → nominal_dt_seconds == 1.0.

    Без target_sample_rate_hz nominal_dt = median положительных diffs реальных
    меток времени. Шаг valve — 1 секунда, поэтому медиана == 1.0. Кроме того, в
    первых 20 строках каждого valve есть ровно один пропущенный секундный отсчёт,
    поэтому report['n_gaps'] >= 1 (ожидается ровно 1, но используем >= для
    устойчивости к возможным правкам данных).
    """
    values, timestamps, _columns = _load_valve_slice(valve)
    _aligned, _ts, _mask, report = align_timestamps(values, timestamps)

    assert report['has_timestamps'] is True
    assert report['nominal_dt_seconds'] == pytest.approx(1.0)
    assert report['n_gaps'] >= 1

# 10. Сквозной prepare_detection_series на реальном срезе valve


@pytest.mark.parametrize('valve', ['valve1', 'valve2'])
def test_prepare_detection_series_smoke_on_valve_slice(valve):
    """prepare_detection_series отрабатывает на реальном срезе valve без падений.

    Прогон с дефолтными политиками (duplicate_policy='keep_last',
    gap_policy='mark_only', без target_sample_rate_hz). Проверяем КОНТРАКТ:
        * aligned_values формы (20, 8) — каналы сохранены;
        * gap_mask — bool длины 20;
        * quality_report.metadata['alignment_report'] присутствует и содержит
          ожидаемые ключи (n_duplicates, n_gaps, gap_total_seconds,
          resampling_method, original_n, aligned_n, has_timestamps,
          nominal_dt_seconds);
        * alignment_report['has_timestamps'] is True, nominal_dt ≈ 1.0;
        * quality_report.n_channels == 8, n_samples == 20.
    Точные числовые значения сигналов НЕ проверяются (нестабильны).
    """
    values, timestamps, _columns = _load_valve_slice(valve)

    aligned_values, gap_mask, quality_report = prepare_detection_series(
        values,
        timestamps=timestamps,
        channel_names=VALVE_FEATURE_COLUMNS,
    )

    assert aligned_values.shape == (VALVE_SLICE_ROWS, VALVE_N_FEATURES)
    assert gap_mask.dtype == bool
    assert gap_mask.shape == (VALVE_SLICE_ROWS,)
    assert quality_report.n_channels == VALVE_N_FEATURES
    assert quality_report.n_samples == VALVE_SLICE_ROWS

    alignment_report = quality_report.metadata['alignment_report']
    expected_keys = {
        'n_duplicates', 'n_gaps', 'gap_total_seconds', 'resampling_method',
        'original_n', 'aligned_n', 'has_timestamps', 'nominal_dt_seconds',
    }
    assert expected_keys.issubset(alignment_report.keys())
    assert alignment_report['has_timestamps'] is True
    assert alignment_report['nominal_dt_seconds'] == pytest.approx(1.0)


@pytest.mark.parametrize('valve', ['valve1', 'valve2'])
def test_prepare_detection_series_resample_forward_fill_no_nans(valve):
    """Ресемплинг valve на 1 Гц с forward_fill не оставляет NaN и сохраняет каналы.

    Прогон с target_sample_rate_hz=1.0 и gap_policy='forward_fill'. Так как в
    первых 20 строках есть пропуск по времени, сетка длиннее 20, и синтетические
    точки заполняются LOCF. Проверяем КОНТРАКТ:
        * число каналов остаётся 8;
        * после forward_fill в aligned_values нет NaN;
        * len(gap_mask) == aligned_values.shape[0] (согласованность длины);
        * gap_mask.sum() >= 1 (есть хотя бы одна синтетическая точка от пропуска);
        * resampling_method == 'forward_fill'.
    """
    values, timestamps, _columns = _load_valve_slice(valve)

    aligned_values, gap_mask, quality_report = prepare_detection_series(
        values,
        timestamps=timestamps,
        target_sample_rate_hz=1.0,
        gap_policy='forward_fill',
        channel_names=VALVE_FEATURE_COLUMNS,
    )

    assert aligned_values.shape[1] == VALVE_N_FEATURES
    assert not np.isnan(aligned_values).any()
    assert len(gap_mask) == aligned_values.shape[0]
    assert gap_mask.sum() >= 1
    assert quality_report.metadata['alignment_report']['resampling_method'] == 'forward_fill'
