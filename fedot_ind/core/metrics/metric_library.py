"""Metric implementations and registry for all task types.

Add new metrics to ``METRIC_REGISTRY`` below (or implement a function and register it).
Each metric callable has the form ``(target, predicted, **params) -> float | list | dict``.
It also can return ``np.ndarray`` ONLY for point-wise metrics for forcasting. The same was in benchmarkV2 in forcasting.py
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional, Tuple, Dict, Union, List, Optional

import numpy as np
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
)

from fedot_ind.core.metrics._exceptions import MetricNotFoundError, MetricValidationError, MetricError


MetricFn = Callable[..., float | list | dict]

# ---------------------------------------------------------------------------
# Registry — register new metrics here
# Dont forget to add it in METRICS_TO_MINIMIZE or METRICS_TO_MAXIMIZE !!!
# ---------------------------------------------------------------------------
METRIC_REGISTRY: dict[str, dict[str, MetricFn]] = {
    'shared_cls_det': {},
    'anomaly_detection': {},
    'shared_reg_forecast': {},
    'forecasting': {},
}

METRICS_TO_MINIMIZE = []
METRICS_TO_MAXIMIZE = []

def get_metric(task: str, name: str) -> MetricFn:
    """Return metric function for *task*."""
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
# A unified metrics module for classification and anomaly detection.
# Supports multi-class and binary cases.
# Easily migrates to PyTorch (see comments).
# ---------------------------------------------------------------------------

# CORE: confusion matrix
def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[Union[np.ndarray, list]] = None,
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

    if len(labels) == 1:
        if labels[0] == 1:
            labels = np.append(labels, 0)
        elif labels[0] == 0:
            labels = np.append(labels, 1)
        else:
            raise MetricValidationError("Please add labels parameter for metric. Default labels for autocomplete is 0 and 1 - integer.")
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

# Public per-class metrics

def per_class_scores(y_true: np.ndarray, y_pred: np.ndarray, *, labels: Union[np.ndarray, list, None] = None, **_) -> Dict[str, list]:
    """
    Возвращает словарь с массивами:
        'recall', 'precision', 'f1', 'support'
    по всем классам (порядок соответствует labels из confusion_matrix).
    """
    cm, _ = confusion_matrix(y_true, y_pred, labels)
    return {
        'recall': _per_class_recall(cm).tolist(),
        'precision': _per_class_precision(cm).tolist(),
        'f1': _per_class_f1(cm).tolist(),
        'support': _per_class_support(cm).tolist()
    }

# Aggregations

def accuracy(y_true: np.ndarray, y_pred: np.ndarray, *, labels: Union[np.ndarray, list, None] = None, **_) -> float:
    """Общая доля правильных ответов."""
    cm, _ = confusion_matrix(y_true, y_pred, labels)
    return float(np.trace(cm) / cm.sum())

def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                      *, labels: Union[np.ndarray, list, None] = None, 
                         pos_label: Union[int, str] = 'auto', 
                         **_) -> float:
    """Balanced accuracy – средний recall по классам."""
    return recall(y_true, y_pred, labels=labels, average='macro', pos_label = pos_label)

# Obtaining a Confusion Matrix for the binary case with a definition of what is the positive class

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

# Binary metrics (precision, recall, f1, FAR, MAR) + components

def binary_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Union[np.ndarray, list, None] = None,
    pos_label: Union[int, str] = 1,
    **_,
) -> Dict[str, int]:
    """Возвращает словарь {'TP': ..., 'TN': ..., 'FP': ..., 'FN': ...}."""
    cm, labels = confusion_matrix(y_true, y_pred, labels)
    tp, tn, fp, fn, _ = _binary_components(cm, labels, pos_label)
    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}

def binary_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Union[np.ndarray, list, None] = None,
    pos_label: Union[int, str] = 1,
    **_,
) -> float:
    return precision(y_true, y_pred, average='binary', labels = labels, pos_label=pos_label)

def binary_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Union[np.ndarray, list, None] = None,
    pos_label: Union[int, str] = 1,
    **_,
) -> float:
    return recall(y_true, y_pred, average='binary', labels=labels, pos_label=pos_label)

def binary_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Union[np.ndarray, list, None] = None,
    pos_label: Union[int, str] = 1,
    **_,
) -> float:
    return f1_score(y_true, y_pred, average='binary', labels=labels, pos_label=pos_label)

def binary_far(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Union[np.ndarray, list, None] = None,
    pos_label: Union[int, str] = 1,
    **_,
) -> float:
    """False Alarm Rate (FAR) в процентах: FP / (FP + TN) * 100."""
    cm, labels = confusion_matrix(y_true, y_pred, labels)
    _, tn, fp, _, _ = _binary_components(cm, labels, pos_label)
    return 100.0 * fp / (fp + tn) if (fp + tn) else 0.0

def binary_mar(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Union[np.ndarray, list, None] = None,
    pos_label: Union[int, str] = 1,
    **_,
) -> float:
    """Miss Alarm Rate (MAR) в процентах: FN / (FN + TP) * 100."""
    cm, labels = confusion_matrix(y_true, y_pred, labels)
    tp, _, _, fn, _ = _binary_components(cm, labels, pos_label)
    return 100.0 * fn / (fn + tp) if (fn + tp) else 0.0

def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Union[np.ndarray, list, None] = None,
    pos_label: Union[int, str] = 1,
    **_,
) -> Dict[str, float]:
    """
    Возвращает словарь:
        TP, TN, FP, FN, precision, recall, f1, FAR (%), MAR (%).
    
            FAR = FP/(FP+TN)*100, MAR = FN/(FN+TP)*100.
    """
    cm_dict = binary_confusion_matrix(y_true, y_pred, labels=labels, pos_label=pos_label)
    return {
        **cm_dict,
        'precision': binary_precision(y_true, y_pred, labels=labels, pos_label=pos_label),
        'recall': binary_recall(y_true, y_pred, labels=labels, pos_label=pos_label),
        'f1': binary_f1(y_true, y_pred, labels=labels, pos_label=pos_label),
        'FAR': binary_far(y_true, y_pred, labels=labels, pos_label=pos_label),
        'MAR': binary_mar(y_true, y_pred, labels=labels, pos_label=pos_label),
    }

# Universal precision/recall/f1 with average selection

def precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Union[np.ndarray, list, None] = None,
    average: str = 'macro',
    pos_label: Union[int, str] = 'auto',
    **_,
) -> float:
    """
    average: 'macro', 'weighted', 'micro', 'binary', 'auto'.
    
    'auto' → macro для >2 классов, иначе binary.
    """
    cm, labels = confusion_matrix(y_true, y_pred, labels=labels)
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
    *,
    labels: Union[np.ndarray, list, None] = None,
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
    cm, labels = confusion_matrix(y_true, y_pred, labels=labels)
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
    *,
    labels: Union[np.ndarray, list, None] = None,
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
    cm, labels = confusion_matrix(y_true, y_pred, labels=labels)
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
        # TODO Хорошо бы сделать логгирование, как описал в ROC-AUC
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
            # TODO Тут бы логгировать, что типа нет данных вероятностей и будет делаться One-Hot для предсказаний и использоваться вместо вероятностей
            y_score = np.zeros((len(predicted), len(classes)))
            for i, c in enumerate(classes):
                y_score[:, i] = (predicted == c).astype(float)
        else:
            y_score = predicted_probs # if predicted_probs.ndim > 1 else encoded
        return float(roc_auc_score(y_true=encoded_target, y_score=y_score, multi_class='ovr', average='macro', labels = classes))
    y_score = predicted_probs.reshape(-1) if predicted_probs.ndim == 1 else predicted_probs[:, 1]
    return float(roc_auc_score(y_true=target, y_score=y_score))

# ---------------------------------------------------------------------------
# Detection — NAB / average_time (binary labels only)
# ---------------------------------------------------------------------------

# Helper (private) functions

def _filter_detecting_boundaries(
    boundaries: List[List[int]]
) -> List[List[int]]:
    """
    Удаляет пустые окна ([]) из списка границ.

    Parameters
    ----------
    boundaries : list of lists
        Список, где каждый элемент либо пустой [], либо пара [left, right].

    Returns
    -------
    list of lists
        Только непустые окна.
    """
    return [b for b in boundaries if len(b) == 2]

def _compute_detecting_boundaries(
    true_array: np.ndarray,
    window_width: Optional[int] = None,
    portion: float = 0.1,
    anomaly_window_destination: str = "lefter",
    intersection_mode: str = "cut right window",
) -> List[List[int]]:
    """
    Строит окна обнаружения вокруг истинных аномалий.

    Parameters
    ----------
    true_array : np.ndarray (1D, бинарный)
        Маска истинных событий (1 = аномалия).
    window_width : int, optional
        Ширина окна в числе отсчётов (индексов). Если None, вычисляется как
        int(len(true_array) / (кол-во аномалий + 1) * portion).
    portion : float
        Используется, если window_width не задан.
    anomaly_window_destination : {'lefter', 'righter', 'center'}
        Положение окна относительно истинной аномалии:
        'lefter' : окно слева от аномалии [val - width, val]
        'righter': окно справа от аномалии [val, val + width]
        'center' : окно симметрично вокруг аномалии.
    intersection_mode : {'cut left window', 'cut right window', 'cut both'}
        Способ разрешения пересекающихся окон (см. документацию оригинального NAB).

    Returns
    -------
    list of lists
        Список окон; каждое окно – [left_index, right_index]. Может быть пустой список,
        если аномалий нет.
    """
    # Индексы истинных аномалий
    true_indices = np.flatnonzero(true_array)
    if len(true_indices) == 0:
        return [[]]

    # TODO ХОРОШО БЫ СДЕЛАТЬ ЛОГГИРОВАНИЕ, ЧТО НЕТ НИ портион, НИ ширины И БУДУТ ИСПОЛЬЗОВАТЬСЯ ДЕФОЛТНЫЕ (ТАКОЕ БЫЛО У ПЕРВОНАЧАЛЬНЫХ АВТОРОВ)

    # Расчёт ширины окна, если не задана явно
    if window_width is None:
        width = int(len(true_array) / (len(true_indices) + 1) * portion)
    else:
        width = window_width

    # Построение начальных границ
    boundaries = []
    for val in true_indices:
        if anomaly_window_destination == "lefter":
            left = val - width 
            right = val
        elif anomaly_window_destination == "righter":
            left = val
            right = val + width
        elif anomaly_window_destination == "center":
            half = width // 2
            left = val - half
            right = val + half
        else:
            raise ValueError(
                "anomaly_window_destination должен быть 'lefter', 'righter' или 'center'"
            )
        boundaries.append([left, right])

    # Разрешение пересечений окон
    if len(boundaries) <= 1:
    # if len(boundaries) == 1:
        return boundaries

    new_boundaries = boundaries.copy()
    for i in range(len(new_boundaries) - 1):
        if new_boundaries[i][1] >= new_boundaries[i + 1][0]:
            if intersection_mode == "cut left window":
                new_boundaries[i][1] = new_boundaries[i + 1][0]
            elif intersection_mode == "cut right window":
                new_boundaries[i + 1][0] = new_boundaries[i][1]
            elif intersection_mode == "cut both":
                a = new_boundaries[i][1]
                new_boundaries[i][1] = new_boundaries[i + 1][0]
                new_boundaries[i + 1][0] = a
            else:
                raise ValueError(
                    "intersection_mode должен быть 'cut left window', "
                    "'cut right window' или 'cut both'"
                )
    return new_boundaries

def _extract_cp_confusion_matrix(
    boundaries: List[List[int]],
    prediction: np.ndarray,
    point: int = 0,
    binary: bool = False,
) -> Dict:
    """
    Извлекает матрицу «окна обнаружения – предсказания».

    Parameters
    ----------
    boundaries : list of lists
        Отфильтрованный список окон [[left, right], ...].
    prediction : np.ndarray (1D, бинарный)
        Предсказания модели.
    point : int
        Индекс срабатывания внутри окна (0 – первое, -1 – последнее) для не-бинарного режима.
    binary : bool
        Если True, в TP сохраняются все предсказания внутри окна.

    Returns
    -------
    dict
        Словарь с ключами:
        'TPs' : dict {номер_окна: [left, позиция_срабатывания, right]}
        'FPs' : np.ndarray индексов ложных срабатываний.
        'FNs' : список номеров окон без предсказаний (или все индексы окна для binary).
    """
    # Индексы всех предсказаний
    times_pred = np.flatnonzero(prediction)
    # Фильтруем границы – оставляем только реальные окна
    boundaries = _filter_detecting_boundaries(boundaries)

    TPs = {}
    FPs = []
    FNs = []

    if len(boundaries) == 0:
        # Все предсказания – ложные
        return {"TPs": TPs, "FPs": times_pred, "FNs": FNs}

    # Обрабатываем промежутки между окнами и внутри них
    for i, (left, right) in enumerate(boundaries):
        # Предсказания, попавшие в окно
        mask = (times_pred >= left) & (times_pred <= right)
        in_window = times_pred[mask]

        if len(in_window) == 0:
            if binary:
                # В binary режиме FN – все индексы внутри окна
                all_indices = np.arange(max(0, left), min(len(prediction), right + 1))
                FNs.append(all_indices)
            else:
                FNs.append(i)
        else:
            if binary:
                TPs[i] = [left, in_window, right]
                # FN – те индексы окна, которые не попали в предсказания
                all_in_window = np.arange(
                    max(0, left), min(len(prediction), right + 1)
                )
                FNs.append(all_in_window[~np.isin(all_in_window, in_window)])
            else:
                idx = in_window[point]  # первое или последнее срабатывание
                TPs[i] = [left, idx, right]

        # Ложные срабатывания между текущим правым и следующим левым окном
        if i < len(boundaries) - 1:
            next_left = boundaries[i + 1][0]
            inter_fp = times_pred[(times_pred > right) & (times_pred < next_left)]
            FPs.append(inter_fp)

    # FPs до первого окна
    first_left = boundaries[0][0]
    FPs.insert(0, times_pred[times_pred < first_left])
    # FPs после последнего окна
    last_right = boundaries[-1][1]
    FPs.append(times_pred[times_pred > last_right])

    # Склеиваем все FPs в один массив
    if FPs:
        FPs = np.concatenate([fp if len(fp) > 0 else np.array([], dtype=int) for fp in FPs])
    else:
        FPs = np.array([], dtype=int)

    if binary:
        if FNs:
            FNs = np.concatenate(FNs) if all(isinstance(f, np.ndarray) for f in FNs) else FNs
        else:
            FNs = np.array([], dtype=int)

    return {"TPs": TPs, "FPs": FPs, "FNs": FNs}

def _my_scale(
    fp_case_window: List[int],
    A_tp: float = 1.0,
    A_fp: float = 0.0,
    koef: float = 1.0,
    detalization: int = 1000,
    clear_anomalies_mode: bool = True,
) -> float:
    """
    Улучшенная оконная функция NAB (tanh-профиль).

    Parameters
    ----------
    fp_case_window : list из трёх int [left, pred_pos, right]
        Координаты окна и позиция предсказания.
    A_tp, A_fp : float
        Веса для истинно-положительного и ложно-положительного исходов.
    koef : float
        Коэффициент крутизны tanh.
    detalization : int
        Число точек дискретизации окна (влияет на точность вычисления скора).
    clear_anomalies_mode : bool
        True – левый край окна = A_tp, правый = A_fp; False – наоборот.

    Returns
    -------
    float
        Взвешенный скор для данного предсказания.
    """
    left, pred_pos, right = fp_case_window
    if right == left:
        return A_fp  # вырожденное окно

    # Положение предсказания в окне в диапазоне [0, 1]
    relative_pos = (pred_pos - left) / (right - left)
    event = int(relative_pos * (detalization - 1))  # переводим в индекс массива x

    x = np.linspace(-np.pi / 2, np.pi / 2, detalization)
    if not clear_anomalies_mode:
        x = x[::-1]

    y = (
        (A_tp - A_fp)
        / 2
        * (-np.tanh(koef * x) / np.tanh(np.pi * koef / 2))
        + (A_tp - A_fp) / 2
        + A_fp
    )

    # Защита от выхода за границы (маловероятно, но возможно при округлении)
    event = min(event, detalization - 1)
    return y[event]

# Main computing functions (private)

def _single_average_delay(
    boundaries: List[List[int]],
    prediction: np.ndarray,
    anomaly_window_destination: str,
    clear_anomalies_mode: bool,
) -> Tuple[int, List[float], int, int]:
    """
    Вычисляет среднюю задержку обнаружения для одного набора границ.

    Возвращает:
        missing      : int – количество пропущенных аномалий
        detectHistory: list[float] – задержки (в числе отсчётов) для обнаруженных аномалий
        FP           : int – количество ложных срабатываний
        all_true_anom: int – общее число истинных аномалий
    """
    boundaries = _filter_detecting_boundaries(boundaries)
    point = 0 if clear_anomalies_mode else -1
    confusion = _extract_cp_confusion_matrix(boundaries, prediction, point=point)

    missing = len(confusion["FNs"])
    FP = len(confusion["FPs"])
    all_true_anom = len(confusion["TPs"]) + missing

    # Выбор формулы задержки в зависимости от положения окна
    if anomaly_window_destination == "lefter":
        def delay(tp_entry): return tp_entry[2] - tp_entry[1]
    elif anomaly_window_destination == "righter":
        def delay(tp_entry): return tp_entry[1] - tp_entry[0]
    elif anomaly_window_destination == "center":
        def delay(tp_entry):
            center = tp_entry[0] + (tp_entry[2] - tp_entry[0]) / 2.0
            return tp_entry[1] - center
    else:
        raise ValueError("anomaly_window_destination должен быть 'lefter', 'righter' или 'center'")

    detectHistory = [delay(confusion["TPs"][i]) for i in confusion["TPs"]]
    return missing, detectHistory, FP, all_true_anom

def _single_evaluate_nab(
    boundaries: List[List[int]],
    prediction: np.ndarray,
    table_of_coef: Optional[Dict] = None,
    clear_anomalies_mode: bool = True,
    scale_func: Callable = _my_scale,
    scale_koef: float = 1.0,
) -> np.ndarray:
    """
    Вычисляет NAB скор для одного набора границ.

    Возвращает np.array формы (3,3):
        [Scores, Scores_null, Scores_perfect] для профилей Standard, LowFP, LowFN.
    """
    boundaries = _filter_detecting_boundaries(boundaries)

    # Таблица коэффициентов по умолчанию
    if table_of_coef is None:
        table_of_coef = {
            "Standard": {"A_tp": 1.0, "A_fp": -0.11, "A_fn": -1.0},
            "LowFP":    {"A_tp": 1.0, "A_fp": -0.22, "A_fn": -1.0},
            "LowFN":    {"A_tp": 1.0, "A_fp": -0.11, "A_fn": -2.0},
        }

    point = 0 if clear_anomalies_mode else -1
    confusion = _extract_cp_confusion_matrix(boundaries, prediction, point=point)

    scores = []
    scores_perfect = []
    scores_null = []

    for profile in ["Standard", "LowFP", "LowFN"]:
        A_tp = table_of_coef[profile]["A_tp"]
        A_fp = table_of_coef[profile]["A_fp"]
        A_fn = table_of_coef[profile]["A_fn"]

        score = 0.0
        score += A_fp * len(confusion["FPs"])
        score += A_fn * len(confusion["FNs"])
        for win_id, tp_entry in confusion["TPs"].items():
            score += scale_func(tp_entry, A_tp, A_fp, koef=scale_koef)

        scores.append(score)
        scores_perfect.append(len(boundaries) * A_tp)
        scores_null.append(len(boundaries) * A_fn)

    return np.array([scores, scores_null, scores_perfect])

# Public API

def compute_nab_score(
    true: np.ndarray,
    prediction: np.ndarray,
    *,
    window_width: Optional[int] = None,
    portion: float = 0.1,
    anomaly_window_destination: str = "lefter",
    clear_anomalies_mode: bool = True,
    intersection_mode: str = "cut right window",
    scale_func: Union[str, Callable] = "my_scale",
    scale_koef: float = 1.0,
    table_of_coef: Optional[Dict] = None,
    **_
) -> Dict[str, float]:
    """
    Вычисление NAB метрики для одной последовательности.

    Parameters
    ----------
    true : np.ndarray (1D, int)
        Бинарная маска истинных аномалий (1 – аномалия, 0 – норма).
    prediction : np.ndarray (1D, int)
        Бинарные предсказания модели.
    window_width : int, optional
        Ширина окна обнаружения в отсчётах. Если None, вычисляется автоматически
        как int(len(prediction) / (число аномалий + 1) * portion).
    portion : float
        Доля длины последовательности, используемая при автоматическом расчёте окна.
    anomaly_window_destination : {'lefter', 'righter', 'center'}
        Положение окна относительно аномалии.
    clear_anomalies_mode : bool
        True – левый край окна соответствует A_tp, правый A_fp.
    intersection_mode : {'cut left window', 'cut right window', 'cut both'}
        Правило разрешения пересекающихся окон.
    scale_func : {'my_scale'} или Callable
        Оконная функция NAB.
        Если 'my_scale' – используется встроенная :func:`_my_scale`.
        Если передан Callable, он должен иметь сигнатуру
        `func(fp_case_window, A_tp, A_fp, koef) -> float`.
    scale_koef : float
        Параметр крутизны для масштабирующей функции.
    table_of_coef : dict, optional
        Профили коэффициентов. По умолчанию – стандартные NAB.

    Returns
    -------
    dict
        {'Standard': float, 'LowFP': float, 'LowFN': float} – NAB скоры в процентах.

    Raises
    ------
    ValueError
        Если истинных аномалий нет (нечего оценивать).
    """
    # Проверка и приведение типов
    true = np.asarray(true, dtype=int).ravel()
    prediction = np.asarray(prediction, dtype=int).ravel()
    if true.shape != prediction.shape:
        raise ValueError("true и prediction должны иметь одинаковую длину")

    # Выбор оконной функции
    if isinstance(scale_func, str):
        if scale_func == "my_scale":
            scale_func_callable = _my_scale
        else:
            raise ValueError(
                "scale_func как строка может быть только 'my_scale'. "
                "Для кастомной функции передайте Callable."
            )
    elif callable(scale_func):
        scale_func_callable = scale_func
    else:
        raise TypeError("scale_func должен быть строкой 'my_scale' или Callable")

    boundaries = _compute_detecting_boundaries(
        true,
        window_width=window_width,
        portion=portion,
        anomaly_window_destination=anomaly_window_destination,
        intersection_mode=intersection_mode,
    )

    # Фильтрованные границы (без пустых)
    filtered = _filter_detecting_boundaries(boundaries)
    if len(filtered) == 0:
        raise ValueError(
            "Не обнаружено ни одной истинной аномалии. "
            "Расчёт NAB невозможен."
        )

    matrix = _single_evaluate_nab(
        filtered,
        prediction,
        table_of_coef=table_of_coef,
        clear_anomalies_mode=clear_anomalies_mode,
        scale_func=scale_func_callable,
        scale_koef=scale_koef,
    )

    results = {}
    for t, profile in enumerate(["Standard", "LowFP", "LowFN"]):
        # Формула NAB: 100 * (Score - Score_null) / (Score_perfect - Score_null)
        num = matrix[0, t] - matrix[1, t]
        den = matrix[2, t] - matrix[1, t]
        # Защита от деления на ноль (теоретически не должно случаться, но для безопасности)
        if den == 0:
            results[profile] = 0.0
        else:
            results[profile] = round(100.0 * num / den, 2)

    return results

def compute_average_time(
    true: np.ndarray,
    prediction: np.ndarray,
    *,
    window_width: Optional[int] = None,
    portion: float = 0.1,
    anomaly_window_destination: str = "lefter",
    clear_anomalies_mode: bool = True,
    intersection_mode: str = "cut right window",
    **_
) -> Dict[str, (float| int)]:
    """
    Вычисление средней задержки обнаружения.

    Parameters
    ----------
    true : np.ndarray (1D, int)
        Бинарная маска истинных аномалий.
    prediction : np.ndarray (1D, int)
        Бинарные предсказания.
    window_width : int, optional
        Ширина окна обнаружения в отсчётах.
    portion : float
        Используется при автоматическом расчёте окна.
    anomaly_window_destination : {'lefter', 'righter', 'center'}
        Положение окна.
    clear_anomalies_mode : bool
        Определяет, какое срабатывание (первое или последнее) считается.
    intersection_mode : str
        Правило разрешения пересечений.

    Returns
    -------
    dict
        {average_delay : float,  # средняя задержка в отсчётах (0, если аномалий не найдено)
         missing : int,          # количество пропущенных аномалий
         FP : int,               # количество ложных срабатываний
         total_anomalies : int}  # общее число истинных аномалий
    """
    true = np.asarray(true, dtype=int).ravel()
    prediction = np.asarray(prediction, dtype=int).ravel()
    if true.shape != prediction.shape:
        raise ValueError("true и prediction должны иметь одинаковую длину")

    boundaries = _compute_detecting_boundaries(
        true,
        window_width=window_width,
        portion=portion,
        anomaly_window_destination=anomaly_window_destination,
        intersection_mode=intersection_mode,
    )

    missing, detect_history, fp, total = _single_average_delay(
        boundaries,
        prediction,
        anomaly_window_destination,
        clear_anomalies_mode,
    )

    avg_delay = float(np.mean(detect_history)) if detect_history else 0.0
    return {
        'average_delay': avg_delay, 
        'missing': missing, 
        'FP': fp, 
        'total_anomalies': total}

def metric_nab_standard(target: np.ndarray, predicted: np.ndarray, **params: Any) -> float:
    return float(compute_nab_score(target, predicted, **params)['Standard'])

def metric_nab_low_fp(target: np.ndarray, predicted: np.ndarray, **params: Any) -> float:
    return float(compute_nab_score(target, predicted, **params)['LowFP'])

def metric_nab_low_fn(target: np.ndarray, predicted: np.ndarray, **params: Any) -> float:
    return float(compute_nab_score(target, predicted, **params)['LowFN'])

# ---------------------------------------------------------------------------
# shared_reg_forecast

# Unified metrics for time series regression and forecasting.
# All implementations are in NumPy, ready for replacement with PyTorch (np -> torch, np.where -> torch.where, etc.).
# ---------------------------------------------------------------------------

# Regression metrics
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
    if (y_true < 0).any() or (y_pred < 0).any():
        raise MetricValidationError('There are negative values in GT or predicted sequence. MSLE does not support negative values')
    return float(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

def mape(y_true: np.ndarray, y_pred: np.ndarray, *, epsilon: float = 1e-8, **_) -> float:
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

# Helper functions for forecasting
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

# Pointwise errors for prediction
def absolute_error(y_true: np.ndarray, y_pred: np.ndarray, **_) -> np.ndarray:
    """Поэлементные абсолютные ошибки."""
    return np.abs(y_true - y_pred)

# Просто так, по сути, пока что не нужна нигде, да и вряд ли понадобится. Я ошибся сделав её, а стирать жалко
def squared_error(y_true: np.ndarray, y_pred: np.ndarray, **_) -> np.ndarray:
    """Поэлементные квадраты ошибок."""
    return (absolute_error(y_true, y_pred) ** 2)
#
    # Pointwise RMSE is identical to absolute_error()

def smape_error(y_true: np.ndarray, y_pred: np.ndarray, *, epsilon: float = 1e-8, **_) -> np.ndarray:
    """
    Поэлементные значения sMAPE (symmetric MAPE), умножаются на 200.
    Возвращает массив той же длины, что и входы.
    """
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < epsilon, epsilon, denom)
    return 100.0 * 2.0 * np.abs(y_pred - y_true) / denom

def mase_error(y_true: np.ndarray, 
               y_pred: np.ndarray,
               *, y_train: np.ndarray, 
                  seasonal_period: int, **_) -> np.ndarray:
    """Поэлементные значения MASE."""
    return absolute_error(y_true, y_pred) / _mase_scale(y_train, seasonal_period)

def owa_error(y_true: np.ndarray, 
              y_pred: np.ndarray,
              *, y_train: np.ndarray, 
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

# Aggregated forecast metrics

    # aggregated mae is identical to mae()

    # aggregated rmse is identical to rmse()

def smape(y_true: np.ndarray, y_pred: np.ndarray, **params) -> float:
    """sMAPE, усреднённый по всем точкам."""
    if params == None:
        params = {}
    epsilon = params.get('epsilon', 1e-8)
    return float(np.mean(smape_error(y_true, y_pred, epsilon=epsilon)))

def mase(y_true: np.ndarray, 
         y_pred: np.ndarray,
         *, y_train: np.ndarray, 
            seasonal_period: int, **_) -> float:
    """MASE, усреднённый по всем точкам."""
    return float(np.mean(mase_error(y_true, y_pred, y_train=y_train, seasonal_period=seasonal_period)))

def owa(y_true: np.ndarray, y_pred: np.ndarray,
        *, y_train: np.ndarray, seasonal_period: int, **params) -> float:
    """
    Агрегированный OWA по всему горизонту:
    0.5 * (smape / smape_baseline + mase / mase_baseline).
    В отличие от среднего от owa_error, здесь агрегация выполняется после усреднения.
    """
    y_train = np.asarray(y_train, dtype=float)
    horizon = len(y_true)
    baseline = _seasonal_naive(y_train, horizon, seasonal_period)

    smape_val = smape(y_true, y_pred, **params)
    mase_val = mase(y_true, y_pred, y_train=y_train, seasonal_period=seasonal_period)
    smape_base = smape(y_true, baseline, **params)
    mase_base = mase(y_true, baseline, y_train=y_train, seasonal_period=seasonal_period)

    smape_base = smape_base if smape_base > 1e-8 else 1.0
    mase_base = mase_base if mase_base > 1e-8 else 1.0

    return float(0.5 * ((smape_val / smape_base) + (mase_val / mase_base)))

# ---------------------------------------------------------------------------
# Populate registry
# ---------------------------------------------------------------------------

def validate_metric_registry(metric_registry):
    """
    Проверяет, что все метрики, зарегистрированные в metric_registry,
    присутствуют либо в METRICS_TO_MINIMIZE, либо в METRICS_TO_MAXIMIZE.

    Args:
        metric_registry: Словарь, где ключи — названия категорий (например, 'shared_cls_det'),
                         значения — словари вида {имя_метрики: функция}.

    Raises:
        ValueError: Если найдены метрики, не попавшие ни в один из кортежей,
                    или если метрика попала одновременно в оба кортежа.
    """
    # Собираем все имена метрик из всех категорий реестра
    all_registered = set()
    for category_dict in metric_registry.values():
        all_registered.update(category_dict.keys())

    covered = set(METRICS_TO_MINIMIZE) | set(METRICS_TO_MAXIMIZE)
    missing_in_minmax = all_registered - covered
    if missing_in_minmax:
        raise MetricError(
            f"Следующие метрики зарегистрированы в реестре, но отсутствуют в кортежах "
            f"METRICS_TO_MINIMIZE или METRICS_TO_MAXIMIZE: {missing_in_minmax}"
        )

    missing_in_reg = covered - all_registered
    if missing_in_reg:
        raise MetricError(
            f"Следующие метрики присутствуют в кортеже METRICS_TO_MINIMIZE или "
            f"METRICS_TO_MAXIMIZE, но отсутствуют в реестре : {missing_in_reg}"
        )

    duplicate = set(METRICS_TO_MINIMIZE) & set(METRICS_TO_MAXIMIZE)
    if duplicate:
        raise MetricError(
            f"Следующие метрики одновременно присутствуют в METRICS_TO_MINIMIZE "
            f"и METRICS_TO_MAXIMIZE: {duplicate}"
        )


def flatten_metric_value(name: str, value: Any) -> dict[str, float]:
    """Развернуть dict-результаты метрик в плоские числовые ключи.

      nab → {'nab.Standard': ..., 'nab.LowFP': ..., 'nab.LowFN': ..., 'nab': nab.Standard}
            (canonical alias под исходным именем для обратной совместимости с
             primary_metric='nab' в presets и FEDOT-обвязке).
      bin_metrics → {'bin_metrics.TP': ..., 'bin_metrics.f1': ..., ..., 'bin_metrics': <first numeric>}
      bin_f1 (float) → {'bin_f1': float}
      per_class_scores (dict со списками) → {} (списки не сводятся скаляром).

    Нечисловые элементы внутри dict пропускаются.
    """
    if isinstance(value, (int, float, np.floating)):
        return {name: float(value)}
    if isinstance(value, dict):
        flat: dict[str, float] = {}
        for sub_key, sub_value in value.items():
            if isinstance(sub_value, (int, float, np.floating)):
                flat[f'{name}.{sub_key}'] = float(sub_value)
        if not flat:
            return {}
        # canonical alias под исходным именем: для nab берём 'Standard', иначе первое числовое
        if f'{name}.Standard' in flat:
            flat.setdefault(name, flat[f'{name}.Standard'])
        else:
            flat.setdefault(name, next(iter(flat.values())))
        return flat
    return {}


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
    METRIC_REGISTRY['anomaly_detection'].update({
    # Unpacking all classification metrics
        **METRIC_REGISTRY['shared_cls_det'],
    # Binary metrics for the case with classes 0 and 1, where 1 is the positive class.
    # THERE IS AN ARGUMENT FOR SETTING pos_label, BUT IT'S HARDCODED FOR 1 NOW
        'bin_confusion_matrix': binary_confusion_matrix,
        'bin_precision': binary_precision,
        'bin_recall': binary_recall,
        'bin_f1': binary_f1,
        'bin_far': binary_far,
        'bin_mar': binary_mar,
        'bin_metrics': binary_metrics,
    # Anomaly Detection-Specific Metrics - NAB and average_time
        'nab': compute_nab_score,
        'nab_standard': metric_nab_standard,
        'nab_low_fp': metric_nab_low_fp,
        'nab_low_fn': metric_nab_low_fn,
        'average_time': compute_average_time,
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
    # Unpacking all regression metrics
        **METRIC_REGISTRY['shared_reg_forecast'],
    # Aggregated, as was done in forecasting
        # 'mae': mae, # есть в регрессии с таким же именем
        # 'rmse': rmse, # есть в регрессии с таким же именем
        'smape': smape,
        'mase': mase,
        'owa': owa,
    # Pointwise, as was done in forecasting
        'pw_mae': absolute_error,
        'pw_rmse': absolute_error,
        'pw_smape': smape_error,
        'pw_mase': mase_error,
        'pw_owa': owa_error,
    })

    # Constants for minimization (the smaller the better)
    global METRICS_TO_MINIMIZE
    METRICS_TO_MINIMIZE = (
        *METRICS_TO_MINIMIZE, 
        'logloss',                # логистическая потеря
        'bin_far',                # доля ложных срабатываний (False Acceptance Rate)
        'bin_mar',                # доля пропущенных тревог (Missed Alarm Rate)
        'average_time',           # среднее время обработки
        'mse',                    # среднеквадратичная ошибка
        'rmse',                   # корень из среднеквадратичной ошибки
        'mae',                    # средняя абсолютная ошибка
        'msle',                   # среднеквадратичная логарифмическая ошибка
        'mape',                   # средняя абсолютная процентная ошибка
        'median_absolute_error',  # медианная абсолютная ошибка
        'max_error',              # максимальная ошибка
        'smape',                  # симметричная MAPE
        'mase',                   # масштабированная абсолютная ошибка
        'owa',                    # общий взвешенный средний (Overall Weighted Average)
        'pw_mae',                 # поточечная MAE
        'pw_rmse',                # поточечная RMSE
        'pw_smape',               # поточечная SMAPE
        'pw_mase',                # поточечная MASE
        'pw_owa',                 # поточечная OWA
    )

    # Constants for maximization (the bigger the better)
    global METRICS_TO_MAXIMIZE
    METRICS_TO_MAXIMIZE = (
        *METRICS_TO_MAXIMIZE,
        'accuracy',
        'balanced_accuracy',
        'f1',
        'precision',
        'recall',
        'roc_auc',
        'per_class_scores',        # сложная метрика (словарь), включается для полноты
        'bin_precision',
        'bin_recall',
        'bin_f1',
        'bin_confusion_matrix',    # матрица ошибок, не скаляр
        'bin_metrics',             # агрегированный словарь метрик
        'nab',                     # Numenta Anomaly Benchmark score (основной)
        'nab_standard',            # NAB со стандартным профилем
        'nab_low_fp',              # NAB с низкой частотой ложных срабатываний
        'nab_low_fn',              # NAB с низкой частотой пропусков
        'r2',                      # коэффициент детерминации R²
        'explained_variance_score',
        'd2_absolute_error_score', # аналог R² для абсолютной ошибки
    )

    validate_metric_registry(METRIC_REGISTRY)

_register_all()
