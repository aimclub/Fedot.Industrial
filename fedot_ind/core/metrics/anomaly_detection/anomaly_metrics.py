import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc


class AnomalyDetectionMetrics:
    """Класс для вычисления метрик качества детектирования аномалий"""

    @staticmethod
    def adjusted_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         contamination: float) -> dict:
        """
        Скорректированные метрики с учетом несбалансированности данных
        """
        from sklearn.metrics import precision_score, recall_score, f1_score

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Adjusted F1 с учетом contamination
        expected_random_f1 = 2 * contamination * (1 - contamination)
        adjusted_f1 = (f1 - expected_random_f1) / (1 - expected_random_f1)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'adjusted_f1': max(0, adjusted_f1)
        }

    @staticmethod
    def range_based_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            tolerance: int = 3) -> dict:
        """
        Метрики, учитывающие временные окна вокруг аномалий
        """

        def find_anomaly_ranges(labels: np.ndarray) -> list:
            ranges = []
            i = 0
            while i < len(labels):
                if labels[i] == 1:
                    start = i
                    while i < len(labels) and labels[i] == 1:
                        i += 1
                    ranges.append((start, i - 1))
                else:
                    i += 1
            return ranges

        true_ranges = find_anomaly_ranges(y_true)
        pred_ranges = find_anomaly_ranges(y_pred)

        # True Positive с учетом tolerance
        tp = 0
        for true_start, true_end in true_ranges:
            detected = False
            for pred_start, pred_end in pred_ranges:
                if (abs(pred_start - true_start) <= tolerance or
                        abs(pred_end - true_end) <= tolerance or
                        (pred_start <= true_start and pred_end >= true_end) or
                        (true_start <= pred_start and true_end >= pred_end)):
                    detected = True
                    break
            if detected:
                tp += 1

        precision_range = tp / len(pred_ranges) if len(pred_ranges) > 0 else 0
        recall_range = tp / len(true_ranges) if len(true_ranges) > 0 else 0
        f1_range = 2 * precision_range * recall_range / (precision_range + recall_range) if (
                                                                                                    precision_range + recall_range) > 0 else 0

        return {
            'range_precision': precision_range,
            'range_recall': recall_range,
            'range_f1': f1_range
        }

    @staticmethod
    def early_detection_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                lead_time: int = 5) -> dict:
        """
        Метрики для оценки раннего обнаружения
        """
        true_anomaly_starts = []
        i = 0
        while i < len(y_true):
            if y_true[i] == 1 and (i == 0 or y_true[i - 1] == 0):
                true_anomaly_starts.append(i)
            i += 1

        early_detections = 0
        for start in true_anomaly_starts:
            # Проверяем, обнаружена ли аномалия заранее
            detection_window = y_pred[max(0, start - lead_time):start + 1]
            if np.any(detection_window == 1):
                early_detections += 1

        early_detection_rate = early_detections / len(true_anomaly_starts) if true_anomaly_starts else 0

        return {
            'early_detection_rate': early_detection_rate,
            'avg_lead_time': lead_time  # Можно усложнить вычисление реального lead time
        }

    @staticmethod
    def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                      scores: np.ndarray, contamination: float) -> dict:
        """
        Комплексная оценка всех метрик
        """
        # Базовые метрики
        base_metrics = AnomalyDetectionMetrics.adjusted_metrics(y_true, y_pred, contamination)

        # Метрики с учетом временных окон
        range_metrics = AnomalyDetectionMetrics.range_based_metrics(y_true, y_pred)

        # Метрики раннего обнаружения
        early_metrics = AnomalyDetectionMetrics.early_detection_metrics(y_true, y_pred)

        # ROC и PR кривые
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(recall, precision)

        # Объединяем все метрики
        comprehensive_metrics = {
            **base_metrics,
            **range_metrics,
            **early_metrics,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'precision_curve': precision.tolist(),
            'recall_curve': recall.tolist()
        }

        return comprehensive_metrics
