# Модуль метрик Fedot Industrial

Единый API для расчёта метрик по **одному** датасету (одна пара target / predicted). Результат — `dict` при наличии флага `return_dataframe=False`; для результата в pandas DataFrame (как раньше в `api/main.py`) используйте те же методы с флагом `return_dataframe=True` или без флага вовсе (дефолтное значение для этого флага `True`).

<!-- [`metrics_implementation.py`](metrics_implementation.py) -->

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
    metrics=('accuracy', 'f1'),
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
  metrics_implementation.py  # legacy-код
  _exceptions.py
  anomaly_detection/function.py  # легаси NAB / average_time (pandas, как в исходном коде) - аналогичная логика перенесена в metrics.py
  README.md
```

## Формат `metric_names`

- строка: `'accuracy'`;
- словарь: `{'name': 'nab', 'params': {'scale_koef': 1.0}}`;
- кортеж/список строк и/или словарей

Повтор одного имени с разными `params` → ключи `nab`, `nab__1`, `nab__2`, …

## Входные данные

Только **последовательности** (списки / numpy), одна серия на вызов. Вложенный список серий `[[0,1], [1,0]]` → `MetricValidationError`.

| Задача | target | predicted_labels | дополнительно |
|--------|--------|------------------|---------------|
| classification | метки классов | `predicted_labels` | `predicted_probs` для logloss / roc_auc |
| detection | 0/1 метки | 0/1 метки | `predicted_probs` и params для NAB / average_time |
| regression | float | float | — |
| forecasting | float горизонт | float горизонт | `train_data`, `seasonality` в kwargs |

`predicted_probs`: 1D (бинарный случай) или 2D `(n_samples, n_classes)`.

## Метрики по задачам

**shared_cls_det** (classification + detection): `accuracy`, `balanced_accuracy`, `f1`, `precision`, `recall`, `per_class_scores` (`dict` с ключами `recall`, `precision`, `f1` `support` и в качестве значений списки метрик по классам), `logloss`, `roc_auc`

**detection**: все метрики из `shared_cls_det`, `bin_confusion_matrix`, `bin_precision`, `bin_recall`, `bin_f1`, `bin_far`, `bin_mar`, `bin_metrics` (`dict` с полями `precision`, `recall`, `f1`, `FAR`, `MAR`), `nab` (`dict` с полями `Standard`; `LowFP`; `LowFN`), `nab_standard`, `nab_low_fp`, `nab_low_fn`, `average_time` (`dict` с полями `average_delay`, `missing`, `FP`, `total_anomalies`)

**shared_reg_forecast** (regression + forecasting): `mse`, `rmse`, `mae`, `r2`, `msle`, `mape`, `median_absolute_error`, `explained_variance_score`, `max_error`, `d2_absolute_error_score`

**forecasting**: все метрики из `shared_reg_forecast`; Аггрегированные: `mae`, `rmse`, `smape`, `mase`, `owa` ; Pointwise → список по горизонту: `pw_mae`, `pw_rmse`, `pw_smape`, `pw_mase`, `pw_owa`

## Добавить метрику

1. Реализовать функцию в [`metric_library.py`](metric_library.py) в нужной секции или импортировать откуда-то.
2. Зарегистрировать в `METRIC_REGISTRY` в начале или конце файла, присвоив новой метрике уникальное имя:
3. Зарегистрировать в `METRICS_TO_MINIMIZE` или `METRICS_TO_MAXIMIZE` в начале или конце файла

Заметка: приятнее регистрировать в начале файла, чтобы было видно, какие именно метрики добавлены, а какие были реализованы по умолчанию в модуле.

## Обратная совместимость

```python
from fedot_ind.core.metrics.metrics_implementation import calculate_forecasting_metric

df = calculate_forecasting_metric(
    target=y_true, predicted_labels=y_pred, metric_names=('rmse',),
    train_data=train, seasonality=12,
)
```

## Исключения

- `MetricValidationError` — неверные входы или отсутствуют probs;
- `MetricNotFoundError` — неизвестное имя метрики.
- `MetricError`

## Тесты

```bash
pytest tests/unit/core/metrics/test_metrics.py -q
```
## Аргументы функций:

### Публичные функции:
- классификация:
    - target
    - predicted_labels
    - predicted_probs = None
    - metric_names = None (выбируться дефолтные 'f1','accuracy')
    - rounding_order = 4
    - return_dataframe = True
    - **kwargs
- детекция:
    - target
    - predicted_labels
    - predicted_probs = None
    - metric_names = None (выбируться дефолтные 'f1','accuracy')
    - rounding_order = 4
    - return_dataframe = True
    - **kwargs
- регрессия:
    - target
    - predicted_labels
    - metric_names = None (выбируться дефолтные 'r2', 'rmse', 'mae')
    - rounding_order = 4
    - return_dataframe = True
    - **kwargs
- прогнозирование:
    - target
    - predicted_labels
    - metric_names = None (выбируться дефолтные 'r2', 'rmse', 'mae')
    - rounding_order = 4
    - return_dataframe = True
    - train_data = None
    - seasonality = None
    - **kwargs

### Особые параметры метрик:
- per_class_scores
    - labels = None (именованный)
- accuracy
    - labels = None (именованный)
- balanced_accuracy
    - labels = None (именованный)
    - pos_label = 'auto' (именованный и будет выбран класс с минимальной поддержкой)
- binary_confusion_matrix / binary_precision / binary_recall / binary_f1 / binary_far / binary_mar / binary_metrics
    - labels = None (именованный)
    - pos_label = 1 (именованный)
- precision / recall / f1_score
    - labels = None (именованный)
    - average = 'auto' (именованный и если классов > 2 → 'weighted', иначе 'binary')
    - pos_label = 'auto' (именованный и для average='binary' – как определить положительный класс)
- metric_logloss / metric_roc_auc
    - predicted_probs = None (именованный и если None, то будет one-hot из predicted_labels)
- NAB
    - window_width: Optional[int] = None,
    - portion: float = 0.1,
    - anomaly_window_destination: str = "lefter",
    - clear_anomalies_mode: bool = True,
    - intersection_mode: str = "cut right window",
    - scale_func: Union[str, Callable] = "my_scale",
    - scale_koef: float = 1.0,
    - table_of_coef: Optional[Dict] = None,

... to be continued...