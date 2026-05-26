"""Metric implementations and registry for all task types.

Add new metrics to ``METRIC_REGISTRY`` below (or implement a function and register it).
Each metric callable has the form ``(target, predicted, **params) -> float | list | dict``.
It also can return ``np.ndarray`` ONLY for point-wise metrics for forcasting. The same was in benchmarkV2 in forcasting.py
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional, Tuple, Dict, Union

import numpy as np
import pandas as pd # НУЖЕН ТОЛЬКО ДЛЯ LEGACY NAB metric
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
)

from fedot_ind.core.metrics._exceptions import MetricNotFoundError, MetricValidationError
from fedot_ind.core.metrics.anomaly_detection.function import (
    single_average_delay,
    single_detecting_boundaries,
    single_evaluate_nab,
)

MetricFn = Callable[..., float | list | dict]

# ---------------------------------------------------------------------------
# Registry — register new metrics here
# ---------------------------------------------------------------------------
METRIC_REGISTRY: dict[str, dict[str, MetricFn]] = {
    'shared_cls_det': {},
    'detection': {},
    'shared_reg_forecast': {},
    'forecasting': {},
}

def get_metric(task: str, name: str) -> MetricFn:
    """Return metric function for *task*; fall back to ``shared``."""
    # Подмена задач, ради обобщения метрик для разных задач
    if task == 'classification':
        taskk = 'shared_cls_det' 
    elif task == 'regression':
        taskk = 'shared_reg_forecast'
    else:
        taskk = task
    # Проверка наличия метрики для такой задачи
    if name in METRIC_REGISTRY.get(taskk, {}):
        return METRIC_REGISTRY[taskk][name]
    raise MetricNotFoundError(f'Unknown metric "{name}" for task "{task}".')

# ---------------------------------------------------------------------------
# shared_cls_det 
# 
# Унифицированный модуль метрик для классификации и обнаружения аномалий.
# Поддерживает многоклассовый и бинарный случаи.
# Легко мигрирует на PyTorch (см. комментарии).
# ---------------------------------------------------------------------------

# ЯДРО: confusion matrix
def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[np.ndarray] = None,
    **_,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет матрицу ошибок NxN.

    Args:
        y_true: истинные метки (1D массив).
        y_pred: предсказанные метки (1D массив).
        labels: явный список меток (1D). Если None, определяются автоматически
                как объединение уникальных значений y_true и y_pred.

    Returns:
        cm: матрица ошибок (N, N), где cm[i, j] – число объектов с истинным классом labels[i]
            и предсказанным labels[j].
        
        labels: отсортированный массив меток, соответствующих осям.
    """
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)
    labels = np.sort(labels)

    # маппинг реальных меток в индексы матрицы
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    idx_true = np.vectorize(label_to_idx.get)(y_true)
    idx_pred = np.vectorize(label_to_idx.get)(y_pred)

    n = len(labels)
    cm = np.zeros((n, n), dtype=np.int64)
    # Для PyTorch:
    # cm = torch.zeros(n, n, dtype=torch.int64)
    # cm.index_put_((idx_true, idx_pred), torch.ones(..., dtype=torch.int64), accumulate=True)
    np.add.at(cm, (idx_true, idx_pred), 1)
    return cm, labels

# Вспомогательные векторные операции над матрицей

def _safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Поэлементное a/b с заменой inf/nan на 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.divide(a, b)
        res[~np.isfinite(res)] = 0.0
    return res

def _per_class_support(cm: np.ndarray) -> np.ndarray:
    """Число истинных объектов каждого класса (сумма по строкам)."""
    return cm.sum(axis=1)

def _per_class_recall(cm: np.ndarray) -> np.ndarray:
    """Recall для каждого класса."""
    tp = cm.diagonal()
    return _safe_divide(tp.astype(float), _per_class_support(cm).astype(float))

def _per_class_precision(cm: np.ndarray) -> np.ndarray:
    """Precision для каждого класса."""
    tp = cm.diagonal()
    predicted_as_class = cm.sum(axis=0)
    return _safe_divide(tp.astype(float), predicted_as_class.astype(float))

def _per_class_f1(cm: np.ndarray) -> np.ndarray:
    """F1-score для каждого класса."""
    prec = _per_class_precision(cm)
    rec = _per_class_recall(cm)
    return _safe_divide(2 * prec * rec, prec + rec)

# Публичные per‑class метрики

def per_class_scores(y_true: np.ndarray, y_pred: np.ndarray, **_) -> Dict[str, list]:
    """
    Возвращает словарь с массивами:
        'recall', 'precision', 'f1', 'support'
    по всем классам (порядок соответствует labels из confusion_matrix).
    """
    cm, _ = confusion_matrix(y_true, y_pred)
    return {
        'recall': _per_class_recall(cm).tolist(),
        'precision': _per_class_precision(cm).tolist(),
        'f1': _per_class_f1(cm).tolist(),
        'support': _per_class_support(cm).tolist()
    }

# Агрегации

def accuracy(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    """Общая доля правильных ответов."""
    cm, _ = confusion_matrix(y_true, y_pred)
    return float(np.trace(cm) / cm.sum())

def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    """Balanced accuracy – средний recall по классам."""
    return recall(y_true, y_pred, average='macro')

# Получение Confusion Matrix для бинарного случая с определением, что является положительным классом

def _binary_components(
    cm: np.ndarray,
    labels: np.ndarray,
    pos_label: Union[int, str] = 'auto'
) -> Tuple[int, int, int, int, int]:
    """
    Извлекает TP, TN, FP, FN и индекс положительного класса из матрицы 2×2.

    Args:
        cm: матрица 2×2.
        labels: метки, соответствующие осям.
        pos_label: значение положительного класса (число) или 'auto' –
                   тогда выбирается класс с наименьшей поддержкой (меньшинство).

    Returns:
        tp, tn, fp, fn, pos_idx
    """
    if cm.shape != (2, 2):
        raise ValueError("Матрица должна быть 2x2 для бинарного случая.")
    if pos_label == 'auto':
        support = _per_class_support(cm)
        pos_idx = int(np.argmin(support))  # класс с наименьшей поддержкой
    else:
        pos_idx_list = np.where(labels == pos_label)[0]
        if len(pos_idx_list) == 0:
            raise ValueError(f"Метка {pos_label} не найдена среди {labels}")
        pos_idx = pos_idx_list[0]
    neg_idx = 1 - pos_idx

    tp = cm[pos_idx, pos_idx]
    tn = cm[neg_idx, neg_idx]
    fp = cm[neg_idx, pos_idx]
    fn = cm[pos_idx, neg_idx]
    return int(tp), int(tn), int(fp), int(fn), pos_idx

# Бинарные метрики (precision, recall, f1, FAR, MAR) + компоненты

def binary_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: Union[int, str] = 1,
    **_,
) -> Dict[str, int]:
    """Возвращает словарь {'TP': ..., 'TN': ..., 'FP': ..., 'FN': ...}."""
    cm, labels = confusion_matrix(y_true, y_pred)
    tp, tn, fp, fn, _ = _binary_components(cm, labels, pos_label)
    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}

def binary_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: Union[int, str] = 1,
    **_,
) -> float:
    return precision(y_true, y_pred, average='binary', pos_label=pos_label)

def binary_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: Union[int, str] = 1,
    **_,
) -> float:
    return recall(y_true, y_pred, average='binary', pos_label=pos_label)

def binary_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: Union[int, str] = 1,
    **_,
) -> float:
    return f1_score(y_true, y_pred, average='binary', pos_label=pos_label)

def binary_far(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: Union[int, str] = 1,
    **_,
) -> float:
    """False Alarm Rate (FAR) в процентах: FP / (FP + TN) * 100."""
    cm, labels = confusion_matrix(y_true, y_pred)
    _, tn, fp, _, _ = _binary_components(cm, labels, pos_label)
    return 100.0 * fp / (fp + tn) if (fp + tn) else 0.0

def binary_mar(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: Union[int, str] = 1,
    **_,
) -> float:
    """Miss Alarm Rate (MAR) в процентах: FN / (FN + TP) * 100."""
    cm, labels = confusion_matrix(y_true, y_pred)
    tp, _, _, fn, _ = _binary_components(cm, labels, pos_label)
    return 100.0 * fn / (fn + tp) if (fn + tp) else 0.0

def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: Union[int, str] = 1,
    **_,
) -> Dict[str, float]:
    """
    Возвращает словарь:
        TP, TN, FP, FN, precision, recall, f1, FAR (%), MAR (%).
    
            FAR = FP/(FP+TN)*100, MAR = FN/(FN+TP)*100.
    """
    cm_dict = binary_confusion_matrix(y_true, y_pred, pos_label)
    return {
        **cm_dict,
        'precision': binary_precision(y_true, y_pred, pos_label),
        'recall': binary_recall(y_true, y_pred, pos_label),
        'f1': binary_f1(y_true, y_pred, pos_label),
        'FAR': binary_far(y_true, y_pred, pos_label),
        'MAR': binary_mar(y_true, y_pred, pos_label),
    }

# Универсальные precision / recall / f1 с выбором average

def precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'macro',
    pos_label: Union[int, str] = 'auto',
    **_,
) -> float:
    """
    average: 'macro', 'weighted', 'micro', 'binary', 'auto'.
    
    'auto' → macro для >2 классов, иначе binary.
    """
    cm, labels = confusion_matrix(y_true, y_pred)
    n_classes  = cm.shape[0]
    if average == 'auto':
        average = 'binary' if n_classes == 2 else 'macro'
    if average == 'binary':
        if n_classes != 2:
            raise ValueError("binary average для precision возможна только при 2 классах")
        tp, _, fp, _, _ = _binary_components(cm, labels, pos_label)
        return float(tp / (tp + fp) if (tp + fp) else 0.0)
    elif average == 'micro':
        # Для precision, recall, F1 в micro всё совпадает с accuracy
        return float(np.trace(cm) / cm.sum()) # micro precision = accuracy
    elif average == 'macro':
        per_class = _per_class_precision(cm)
        return float(np.mean(per_class))
    elif average == 'weighted':
        """Средневзвешенное по поддержке классов."""
        per_class = _per_class_precision(cm)
        support = _per_class_support(cm)
        return float(np.average(per_class, weights=support))
    else:
        raise ValueError(f"Неизвестный тип усреднения: {average}")

def recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'macro',
    pos_label: Union[int, str] = 'auto',
    **_,
) -> float:
    """
    recall с выбором способа усреднения.

    Args:
        average: 'macro', 'weighted', 'micro', 'binary', 'auto'.
        
                'auto' → macro для >2 классов, иначе binary.

    pos_label: для average='binary' – как определить положительный класс.
    """
    cm, labels = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    if average == 'auto':
        average = 'binary' if n_classes == 2 else 'macro'
    if average == 'binary':
        if n_classes != 2:
            raise ValueError("binary average для recall возможна только при 2 классах")
        tp, _, _, fn, _ = _binary_components(cm, labels, pos_label)
        return float(tp / (tp + fn) if (tp + fn) else 0.0)
    elif average == 'micro':
        # Для precision, recall, F1 в micro всё совпадает с accuracy
        return float(np.trace(cm) / cm.sum())  # micro recall = accuracy
    elif average == 'macro':
        per_class = _per_class_recall(cm)
        return float(np.mean(per_class))
    elif average == 'weighted':
        per_class = _per_class_recall(cm)
        support = _per_class_support(cm)
        return float(np.average(per_class, weights=support))
    else:
        raise ValueError(f"Неизвестный тип усреднения: {average}")

def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'auto',
    pos_label: Union[int, str] = 'auto',
    **_,
) -> float:
    """
    F1-score с выбором способа усреднения.

    Args:
        average: 'macro', 'weighted', 'micro', 'binary', 'auto'.
                 
                 'auto': если классов > 2 → 'weighted', иначе 'binary'.
        
        pos_label: для average='binary' – как определить положительный класс.
    """
    cm, labels = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    if average == 'auto':
        average = 'binary' if n_classes == 2 else 'weighted'

    if average == 'binary':
        if n_classes != 2:
            raise ValueError("binary average для F1 возможна только при 2 классах")
        tp, _, fp, fn, _ = _binary_components(cm, labels, pos_label)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return float(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    elif average == 'micro':
        # Для precision, recall, F1 в micro всё совпадает с accuracy
        return float(np.trace(cm) / cm.sum()) # micro recall = accuracy
    elif average == 'macro':
        per_class = _per_class_f1(cm)
        return float(np.mean(per_class))
    elif average == 'weighted':
        per_class = _per_class_f1(cm)
        support = _per_class_support(cm)
        return float(np.average(per_class, weights=support))
    else:
        raise ValueError(f"Неизвестный тип усреднения: {average}")

# Остатки от классификации (может быть уже легаси)

def metric_logloss(target: np.ndarray, predicted: np.ndarray, *, predicted_probs=None, **_) -> float:
    if predicted_probs is None and predicted is None:
        raise MetricValidationError('logloss requires predicted_probs. Or at least predicted (one-hot and get similar at probs)')
    if predicted_probs is None and predicted is not None:
        classes = sorted(set(int(v) for v in np.unique(target)) | set(int(v) for v in np.unique(predicted)))
        predicted_probs = np.zeros((len(predicted), len(classes)))
        for i, c in enumerate(classes):
            predicted_probs[:, i] = (predicted == c).astype(float)
    return float(log_loss(y_true=target, y_pred=predicted_probs))

def metric_roc_auc(target: np.ndarray, predicted: np.ndarray, *, predicted_probs=None, **_) -> float:
    # if predicted_probs is None:
    #     raise MetricValidationError('roc_auc requires predicted_probs.')
    # classes = np.unique(target)
    classes = sorted(set(int(v) for v in np.unique(target)) | set(int(v) for v in np.unique(predicted)))
    if len(classes) > 2:
        encoded_target = np.zeros((len(target), len(classes)))
        for i, c in enumerate(classes):
            encoded_target[:, i] = (target == c).astype(float)
        if predicted_probs is None and predicted is not None:
            y_score = np.zeros((len(predicted), len(classes)))
            for i, c in enumerate(classes):
                y_score[:, i] = (predicted == c).astype(float)
        else:
            y_score = predicted_probs # if predicted_probs.ndim > 1 else encoded
        return float(roc_auc_score(y_true=encoded_target, y_score=y_score, multi_class='ovr', average='macro', labels = classes))
    y_score = predicted_probs.reshape(-1) if predicted_probs.ndim == 1 else predicted_probs[:, 1]
    return float(roc_auc_score(y_true=target, y_score=y_score))

# ---------------------------------------------------------------------------
# Detection — NAB / average_time (legacy pandas helpers, binary labels only)
# ---------------------------------------------------------------------------

def _labels_to_series(values: np.ndarray) -> pd.Series:
    array = np.asarray(values, dtype=int).reshape(-1)
    return pd.Series(array, index=pd.RangeIndex(len(array)))


def _detecting_boundaries_from_binary(
        target: np.ndarray,
        predicted: np.ndarray,
        **params: Any,
) -> list[list]:
    target_series = _labels_to_series(target)
    boundaries = single_detecting_boundaries(
        target_series=target_series,
        target_list_ts=None,
        predicted_labels=_labels_to_series(predicted),
        share=float(params.get('share', 0.1)),
        window_width=params.get('window_width', 1),
        anomaly_window_destination=str(params.get('anomaly_window_destination', 'lefter')),
        intersection_mode=str(params.get('intersection_mode', 'cut right window')),
    )
    return boundaries


def metric_nab(target: np.ndarray, predicted: np.ndarray, **params: Any) -> float | dict[str, float]:
    boundaries = _detecting_boundaries_from_binary(target, predicted, **params)
    matrix = single_evaluate_nab(
        boundaries,
        _labels_to_series(predicted),
        table_of_val=params.get('table_of_val'),
        clear_anomalies_mode=bool(params.get('clear_anomalies_mode', True)),
        scale_val=float(params.get('scale_val', 1.0)),
    )
    profile = str(params.get('profile', 'all'))
    names = ['Standard', 'LowFP', 'LowFN']
    results = {}
    for t, profile_name in enumerate(names):
        val = round(100 * (matrix[0, t] - matrix[1, t]) / (matrix[2, t] - matrix[1, t]), 2)
        results[profile_name] = val
    if profile != 'all':
        return results[profile]
    return results


def metric_average_time(target: np.ndarray, predicted: np.ndarray, **params: Any) -> float | dict[str, Any]:
    boundaries = _detecting_boundaries_from_binary(target, predicted, **params)
    missing, detect_history, fp, all_target = single_average_delay(
        boundaries,
        _labels_to_series(predicted),
        anomaly_window_destination=str(params.get('anomaly_window_destination', 'lefter')),
        clear_anomalies_mode=bool(params.get('clear_anomalies_mode', True)),
    )
    # mean_delay = float(np.mean(detect_history)) if detect_history else 0.0
    if detect_history:
        delays = [
            float(item.total_seconds()) if hasattr(item, 'total_seconds') else float(item)
            for item in detect_history
        ]
        mean_delay = float(np.mean(delays))
    else:
        mean_delay = 0.0
    if params.get('return_breakdown'):
        return {
            'false_positive': int(fp),
            'missed_anomaly': int(missing),
            'all_detection_hist': mean_delay,
            'all_anomaly_history': int(all_target),
        }
    return mean_delay


def metric_nab_standard(target: np.ndarray, predicted: np.ndarray, **params: Any) -> float:
    return float(metric_nab(target, predicted, profile='Standard', **params))


def metric_nab_low_fp(target: np.ndarray, predicted: np.ndarray, **params: Any) -> float:
    return float(metric_nab(target, predicted, profile='LowFP', **params))


def metric_nab_low_fn(target: np.ndarray, predicted: np.ndarray, **params: Any) -> float:
    return float(metric_nab(target, predicted, profile='LowFN', **params))


# ---------------------------------------------------------------------------
# shared_reg_forecast

# Унифицированные метрики для регрессии и прогнозирования временных рядов.
# Все реализации на NumPy, готовы к замене на PyTorch (np -> torch, np.where -> torch.where и т.д.).
# ---------------------------------------------------------------------------

# Регрессионные метрики
def mse(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))

def rmse(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))

def mae(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    """R² (coefficient of determination)."""
    residual = np.sum((y_true - y_pred) ** 2)
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - residual / total) if total > 1e-12 else 0.0

def msle(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    """Mean Squared Logarithmic Error."""
    return float(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8, **_) -> float:
    """
    Mean Absolute Percentage Error (возвращает долю, не проценты).
    Совпадает с поведением sklearn.metrics.mean_absolute_percentage_error.
    """
    denom = np.maximum(epsilon, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)))

def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    """Median Absolute Error."""
    return float(np.median(np.abs(y_true - y_pred)))

def explained_variance_score(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    """Explained variance regression score."""
    diff_var = np.var(y_true - y_pred)
    y_var = np.var(y_true)
    return float(1.0 - diff_var / y_var) if y_var > 1e-12 else 0.0

def max_error(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    """Max residual error."""
    return float(np.max(np.abs(y_true - y_pred)))

def d2_absolute_error_score(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    """
    D² regression score for absolute error (аналог R² для MAE).
    """
    numerator = mae(y_true, y_pred)
    denominator = mae(y_true, np.median(y_true))
    return float(1.0 - numerator / denominator) if denominator > 1e-12 else 0.0

# ---------------------------------------------------------------------------
# Forecasting (from benchmark/v2/forecasting.py)
# ---------------------------------------------------------------------------

# Вспомогательные функции для прогнозирования
def _mase_scale(train: np.ndarray, seasonal_period: int) -> float:
    """
    Масштаб для MASE: среднее абсолютное изменение ряда с лагом seasonal_period (или 1).
    """
    train = np.asarray(train, dtype=float).reshape(-1) # типа безопаснее было бы сделать .flatten() {не меняет область памяти, а создаёт копию}, но в исходном было reshape(-1), значит так и оставлю
    lag = seasonal_period if seasonal_period > 1 and len(train) > seasonal_period else 1
    if len(train) <= lag:
        return 1.0
    scale = float(np.mean(np.abs(train[lag:] - train[:-lag])))
    return scale if scale > 1e-8 else 1.0

def _seasonal_naive(train: np.ndarray, horizon: int, seasonal_period: int) -> np.ndarray:
    """
    Сезонный наивный прогноз: повтор последнего сезонного блока нужной длины (или 1).
    """
    train = np.asarray(train, dtype=float).reshape(-1)
    lag = seasonal_period if seasonal_period > 1 and len(train) > seasonal_period else 1
    base = train[-lag:]
    repeats = int(np.ceil(horizon / lag))
    return np.tile(base, repeats)[:horizon]

# Поэлементные (pointwise) ошибки для прогнозирования
def absolute_error(y_true: np.ndarray, y_pred: np.ndarray, **_) -> np.ndarray:
    """Поэлементные абсолютные ошибки."""
    return np.abs(y_true - y_pred)

# Просто так, по сути, пока что не нужна нигде, да и вряд ли понадобится. Я ошибся сделав её, а стирать жалко
def squared_error(y_true: np.ndarray, y_pred: np.ndarray, **_) -> np.ndarray:
    """Поэлементные квадраты ошибок."""
    return (absolute_error(y_true, y_pred) ** 2)
#
    # поэлементное RMSE тождественно absolute_error()

def smape_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8, **_) -> np.ndarray:
    """
    Поэлементные значения sMAPE (symmetric MAPE), умножаются на 200.
    Возвращает массив той же длины, что и входы.
    """
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < epsilon, epsilon, denom)
    return 100.0 * 2.0 * np.abs(y_pred - y_true) / denom

def mase_error(y_true: np.ndarray, 
               y_pred: np.ndarray,
               y_train: np.ndarray, 
               seasonal_period: int, **_) -> np.ndarray:
    """Поэлементные значения MASE."""
    return absolute_error(y_true, y_pred) / _mase_scale(y_train, seasonal_period)

def owa_error(y_true: np.ndarray, 
              y_pred: np.ndarray,
              y_train: np.ndarray, 
              seasonal_period: int, **_) -> np.ndarray:
    """
    Поэлементные значения OWA (Overall Weighted Average).
    Возвращает массив, каждый элемент которого вычислен по формуле
    0.5 * (smape_i / smape_baseline_i + mase_i / mase_baseline_i).
    """
    y_train = np.asarray(y_train, dtype=float)
    horizon = len(y_true)
    baseline = _seasonal_naive(y_train, horizon, seasonal_period)

    smape_actual = smape_error(y_true, y_pred)
    mase_actual = mase_error(y_true, y_pred, y_train, seasonal_period)
    smape_base = smape_error(y_true, baseline)
    mase_base = mase_error(y_true, baseline, y_train, seasonal_period)

    # защита от деления на 0
    smape_base = np.where(smape_base < 1e-8, 1.0, smape_base)
    mase_base = np.where(mase_base < 1e-8, 1.0, mase_base)

    return 0.5 * ((smape_actual / smape_base) + (mase_actual / mase_base))

# Агрегированные прогнозные метрики

    # агрегированная mae тождественна mae()

    # агрегированная rmse тождественна rmse()

def smape(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    """sMAPE, усреднённый по всем точкам."""
    return float(np.mean(smape_error(y_true, y_pred)))

def mase(y_true: np.ndarray, 
         y_pred: np.ndarray,
         y_train: np.ndarray, 
         seasonal_period: int, **_) -> float:
    """MASE, усреднённый по всем точкам."""
    return float(np.mean(mase_error(y_true, y_pred, y_train, seasonal_period)))

def owa(y_true: np.ndarray, y_pred: np.ndarray,
        y_train: np.ndarray, seasonal_period: int, **_) -> float:
    """
    Агрегированный OWA по всему горизонту:
    0.5 * (smape / smape_baseline + mase / mase_baseline).
    В отличие от среднего от owa_error, здесь агрегация выполняется после усреднения.
    """
    y_train = np.asarray(y_train, dtype=float)
    horizon = len(y_true)
    baseline = _seasonal_naive(y_train, horizon, seasonal_period)

    smape_val = smape(y_true, y_pred)
    mase_val = mase(y_true, y_pred, y_train, seasonal_period)
    smape_base = smape(y_true, baseline)
    mase_base = mase(y_true, baseline, y_train, seasonal_period)

    smape_base = smape_base if smape_base > 1e-8 else 1.0
    mase_base = mase_base if mase_base > 1e-8 else 1.0

    return float(0.5 * ((smape_val / smape_base) + (mase_val / mase_base)))

# ---------------------------------------------------------------------------
# Populate registry
# ---------------------------------------------------------------------------

def _register_all() -> None:
    METRIC_REGISTRY['shared_cls_det'].update({
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1': f1_score,
        'precision': precision,
        'recall': recall,
        'per_class_scores': per_class_scores,
        'logloss': metric_logloss,
        'roc_auc': metric_roc_auc,
    })
    METRIC_REGISTRY['detection'].update({
    # Распаковка всех метрик классификации
        **METRIC_REGISTRY['shared_cls_det'],
    # Бинарные метрики для случая с классами 0 и 1, где 1 - положительный класс. 
    # ЕСТЬ АРГУМЕНТ ДЛЯ ЗАДАНИЯ pos_label, НО ЩАС ХАРДКОД НА 1
        'bin_confusion_matrix': binary_confusion_matrix,
        'bin_precision': binary_precision,
        'bin_recall': binary_recall,
        'bin_f1': binary_f1,
        'bin_far': binary_far,
        'bin_mar': binary_mar,
        'bin_metrics': binary_metrics,
    # Специфичные для Обнаружения Аномалий метрики - NAB и average_time
        'nab': metric_nab,
        'nab_standard': metric_nab_standard,
        'nab_low_fp': metric_nab_low_fp,
        'nab_low_fn': metric_nab_low_fn,
        'average_time': metric_average_time,
    })
    METRIC_REGISTRY['shared_reg_forecast'].update({
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2_score,
        'msle': msle,
        'mape': mape,
        'median_absolute_error': median_absolute_error,
        'explained_variance_score': explained_variance_score,
        'max_error': max_error,
        'd2_absolute_error_score': d2_absolute_error_score,
    })
    METRIC_REGISTRY['forecasting'].update({
    # Распаковка всех метрик регрессии
        **METRIC_REGISTRY['shared_reg_forecast'],
    # Аггрегированные, как было сделано в forcasting
        # 'mae': mae, # есть в регрессии с таким же именем
        # 'rmse': rmse, # есть в регрессии с таким же именем
        'smape': smape,
        'mase': mase,
        'owa': owa,
    # Поточечные, как было сделано в forcasting
        'pw_mae': absolute_error,
        'pw_rmse': absolute_error,
        'pw_smape': smape_error,
        'pw_mase': mase_error,
        'pw_owa': owa_error,
    })

_register_all()
