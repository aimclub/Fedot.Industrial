# Модуль метрик Fedot Industrial

Единый API для расчёта метрик по **одному** датасету (одна пара target / predicted). Результат — `dict`; для pandas DataFrame используйте [`metrics_implementation.py`](metrics_implementation.py) (как раньше в `api/main.py`).

## Быстрый старт

```python
from fedot_ind.core.metrics.metrics import (
    calculate_classification_metric,
    calculate_detection_metric,
    calculate_regression_metric,
    calculate_forecasting_metric,
)

scores = calculate_classification_metric(
    target=[0, 1, 1, 0],
    predicted_labels=[0, 1, 0, 0],
    metrics=('accuracy', 'f1'),
)

scores = calculate_detection_metric(
    target=[0, 0, 1, 1, 0],
    predicted_labels=[0, 0, 1, 0, 0],
    metrics=('accuracy', 'f1_macro'),
)

scores = calculate_regression_metric(
    target=[1.0, 2.0, 3.0],
    predicted=[1.1, 2.0, 2.9],
    metrics=('r2', 'rmse'),
)

scores = calculate_forecasting_metric(
    target=actual,
    predicted=forecast,
    metrics=[{'name': 'smape'}, {'name': 'mase'}],
    train_data=history,
    seasonality=12,
)
```

## Структура

```
metrics/
  metrics.py              # QualityMetric, calculate_* (dict)
  metric_library.py       # все реализации + METRIC_REGISTRY
  metrics_implementation.py  # legacy-классы F1/RMSE, DataFrame-обёртки
  _exceptions.py
  anomaly_detection/function.py  # NAB / average_time (pandas, как в исходном коде)
  README.md
```

## Формат `metrics`

- строка: `'accuracy'`;
- словарь: `{'name': 'nab', 'params': {'scale_val': 1.0}}`.

Повтор одного имени с разными `params` → ключи `nab`, `nab__1`, `nab__2`, …

## Входные данные

Только **последовательности** (списки / numpy), одна серия на вызов. Вложенный список серий `[[0,1], [1,0]]` → `MetricValidationError`.

| Задача | target | predicted | дополнительно |
|--------|--------|-----------|----------------|
| classification | метки классов | `predicted_labels` | `predicted_probs` для logloss / roc_auc |
| regression | float | float | — |
| forecasting | float горизонт | float горизонт | `train_data`, `seasonality` в kwargs |
| detection | 0/1 метки | 0/1 метки | params для NAB / average_time |

`predicted_probs`: 1D (бинарный случай) или 2D `(n_samples, n_classes)`.

## Метрики по задачам

**shared** (classification + detection): `accuracy`, `balanced_accuracy`, `f1_macro`, `f1`, `precision`, `recall`, `confusion_matrix`, `binary`

**classification**: `logloss`, `roc_auc`

**regression**: `r2`, `mse`, `rmse`, `mae`, `msle`, `mape`, `median_absolute_error`, `explained_variance_score`, `max_error`, `d2_absolute_error_score`

**forecasting**: `mae`, `rmse`, `smape`, `mape`, `mdae`, `mase`, `mdase`, `owa` — param `pointwise: True` → список по горизонту

**detection**: `nab` (`profile`: `all` | `Standard` | `LowFP` | `LowFN`), `nab_standard`, `nab_low_fp`, `nab_low_fn`, `average_time` (`return_breakdown: True` для полного словаря)

## Добавить метрику

1. Реализовать функцию в [`metric_library.py`](metric_library.py) в нужной секции.
2. Зарегистрировать в `METRIC_REGISTRY` в `_register_all()`:

```python
METRIC_REGISTRY['regression']['my_metric'] = my_metric_fn
```

## Обратная совместимость

```python
from fedot_ind.core.metrics.metrics_implementation import calculate_forecasting_metric

df = calculate_forecasting_metric(
    target=y_true, labels=y_pred, metric_names=('rmse',),
    train_data=train, seasonality=12,
)
```

## Исключения

- `MetricValidationError` — неверные входы или отсутствуют probs;
- `MetricNotFoundError` — неизвестное имя метрики.

## Тесты

```bash
pytest tests/unit/core/metrics/test_metrics.py -q
```
