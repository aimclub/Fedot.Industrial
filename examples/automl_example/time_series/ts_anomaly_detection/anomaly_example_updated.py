import matplotlib.pyplot as plt
import numpy as np

from fedot_ind.core.architecture.preprocessing.data_splitter import DataSplitter
from fedot_ind.core.metrics.anomaly_detection.anomaly_metrics import AnomalyDetectionMetrics
from fedot_ind.core.models.detection.shapelet_model import OptimizedShapeletAnomalyDetector
from fedot_ind.tools.synthetic.anomaly_generator import generate_univariate_anomaly_data, \
    generate_multivariate_anomaly_data, generate_online_anomaly_data

# Тестирование разных детекторов
DEFAULT_DETECTOR_FOR_UNI = {
    # 'PrefixBased': PrefixLengthAnomalyDetector(),
    # 'StateTransition': StateTransitionAnomalyDetector(),
    # 'ShapeletBased': ShapeletAnomalyDetector(),
    'OptimizedShapelet': OptimizedShapeletAnomalyDetector(),
    # 'Autoencoder': AutoencoderAnomalyDetector(contamination=0.05, n_epochs=50),
    # 'LSTM': LSTMAnomalyDetector(contamination=0.05, n_epochs=50),
    # 'Hybrid': HybridStatisticalDLDetector(contamination=0.05, n_epochs=50),
    # 'Ensemble': DynamicEnsembleDetector(contamination=0.05)
}

# Детекторы, работающие с многомерными данными
DEFAULT_DETECTOR_FOR_MULTI = {
    # 'PrefixBased_MV': PrefixLengthAnomalyDetector(contamination=0.05),
    # 'Autoencoder_MV': AutoencoderAnomalyDetector(contamination=0.05, n_epochs=50),
    # 'LSTM_MV': LSTMAnomalyDetector(contamination=0.05, n_epochs=50),
    # 'Hybrid_MV': HybridStatisticalDLDetector(contamination=0.05, n_epochs=50),
    # 'Ensemble_MV': DynamicEnsembleDetector(contamination=0.05)
}

# Онлайн-детекторы
DEFAULT_ONLINE_DETECTOR = {
    # 'OnlineAutoencoder': OnlineAutoencoderDetector(contamination=0.05, window_size=200),
    # 'AdaptiveThreshold': AdaptiveThresholdDetector(contamination=0.05, window_size=200)
}


class AnomalyDetectionDemo:
    """Класс с примерами использования для разных типов данных"""

    @staticmethod
    def demo_univariate():
        """Демонстрация для одномерных данных"""
        print("=== ДЕМОНСТРАЦИЯ: Одномерные временные ряды ===")

        # Генерация данных
        input_data = generate_univariate_anomaly_data(n_samples=500, n_anomalies=25)
        splitter = DataSplitter(test_size=0.3, temporal_split=True)
        input_data_train, input_data_test = splitter.split(input_data)
        print(f"Данные: {input_data.features.shape}, Аномалий: {np.sum(input_data.target)}")

        # Тестирование разных детекторов
        detectors = DEFAULT_DETECTOR_FOR_UNI

        results = {}
        for name, detector in detectors.items():
            try:
                # Обучение и предсказание
                detector.fit(input_data_train)
                y_pred = detector.predict(input_data_test)
                scores = detector.predict_proba(input_data_train)

                # Вычисление метрик
                metrics = AnomalyDetectionMetrics.compute_comprehensive_metrics(
                    input_data.target, y_pred, scores, contamination=0.05
                )

                results[name] = {
                    'predictions': y_pred,
                    'scores': scores,
                    'metrics': metrics
                }

                print(f"\n{name}:")
                print(
                    f"  F1: {metrics['f1']:.3f}, Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
                print(f"  ROC-AUC: {metrics['roc_auc']:.3f}, PR-AUC: {metrics['pr_auc']:.3f}")

            except Exception as e:
                print(f"Ошибка в {name}: {e}")

        # Визуализация
        AnomalyDetectionDemo._plot_univariate_results(input_data, results)

        return results

    @staticmethod
    def demo_multivariate():
        """Демонстрация для многомерных данных"""
        print("\n=== ДЕМОНСТРАЦИЯ: Многомерные временные ряды ===")

        # Генерация данных
        X_multivariate, y_true = generate_multivariate_anomaly_data(n_samples=800, n_features=3, n_anomalies=40)
        print(f"Данные: {X_multivariate.shape}, Аномалий: {np.sum(y_true)}")

        detectors = DEFAULT_DETECTOR_FOR_MULTI

        results = {}
        for name, detector in detectors.items():
            try:
                # Обучение и предсказание
                y_pred = detector.fit(X_multivariate, y_true)
                scores = detector.decision_function(X_multivariate)

                # Метрики
                metrics = AnomalyDetectionMetrics.compute_comprehensive_metrics(
                    y_true, y_pred, scores, contamination=0.05
                )

                results[name] = {
                    'predictions': y_pred,
                    'scores': scores,
                    'metrics': metrics
                }

                print(f"\n{name}:")
                print(f"  F1: {metrics['f1']:.3f}, Adjusted F1: {metrics['adjusted_f1']:.3f}")
                print(f"  Range F1: {metrics['range_f1']:.3f}, Early Detection: {metrics['early_detection_rate']:.3f}")

            except Exception as e:
                print(f"Ошибка в {name}: {e}")

        # Визуализация
        AnomalyDetectionDemo._plot_multivariate_results(X_multivariate, y_true, results)

        return results

    @staticmethod
    def demo_online_learning():
        """Демонстрация онлайн-обучения"""
        print("\n=== ДЕМОНСТРАЦИЯ: Онлайн-обучение ===")

        online_detectors = DEFAULT_ONLINE_DETECTOR
        X_online = generate_online_anomaly_data()
        # Симуляция потокового обучения
        batch_size = 50
        all_predictions = {}
        all_scores = {}

        for name, detector in online_detectors.items():
            predictions = []
            scores = []

            for i in range(0, len(X_online), batch_size):
                batch = X_online[i:i + batch_size]

                # Инкрементальное обучение
                detector.partial_fit(batch)

                # Предсказание на текущем батче
                batch_scores = detector.decision_function(batch)
                batch_pred = detector.predict(batch)

                scores.extend(batch_scores)
                predictions.extend(batch_pred)

            all_predictions[name] = np.array(predictions)
            all_scores[name] = np.array(scores)

            print(f"\n{name}:")
            print(f"  Обучено на {len(X_online)} точках в режиме онлайн")
            print(f"  Размер окна: {detector.window_size}")

        # Визуализация онлайн-обучения
        AnomalyDetectionDemo._plot_online_results(X_online, all_predictions, all_scores)

        return all_predictions, all_scores

    @staticmethod
    def _plot_univariate_results(input_data, results):
        """Визуализация результатов для одномерных данных"""
        X, y_true = input_data.features, input_data.target
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # Исходные данные с аномалиями
        axes[0].plot(X, 'b-', alpha=0.7, label='Данные')
        anomalies_idx = np.where(y_true == 1)[0]
        axes[0].scatter(anomalies_idx, X[anomalies_idx], color='red',
                        label='Истинные аномалии', zorder=5)
        axes[0].set_title('Исходные данные с аномалиями')
        axes[0].legend()

        # Предсказания каждого детектора
        for idx, (name, result) in enumerate(results.items()):
            if idx + 1 >= len(axes):
                break

            ax = axes[idx + 1]
            ax.plot(X, 'b-', alpha=0.3)
            pred_anomalies = np.where(result['predictions'] == 1)[0]
            ax.scatter(pred_anomalies, X[pred_anomalies], color='orange',
                       label='Предсказанные аномалии', zorder=5)
            ax.set_title(f'{name}\nF1: {result["metrics"]["f1"]:.3f}')
            ax.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _plot_multivariate_results(X, y_true, results):
        """Визуализация результатов для многомерных данных"""
        n_features = X.shape[1]
        fig, axes = plt.subplots(n_features + 1, 1, figsize=(12, 3 * (n_features + 1)))

        if n_features == 1:
            axes = [axes]

        # Исходные данные по каждому измерению
        for i in range(n_features):
            axes[i].plot(X[:, i], 'b-', alpha=0.7, label=f'Измерение {i + 1}')
            anomalies_idx = np.where(y_true == 1)[0]
            axes[i].scatter(anomalies_idx, X[anomalies_idx, i], color='red',
                            label='Аномалии', zorder=5)
            axes[i].legend()
            axes[i].set_ylabel(f'Dim {i + 1}')

        # Сравнение метрик
        metrics_names = ['f1', 'adjusted_f1', 'range_f1', 'early_detection_rate']
        metrics_values = {name: [] for name in metrics_names}
        detector_names = list(results.keys())

        for detector in detector_names:
            metrics = results[detector]['metrics']
            for metric_name in metrics_names:
                metrics_values[metric_name].append(metrics.get(metric_name, 0))

        x = np.arange(len(detector_names))
        width = 0.2

        for idx, metric_name in enumerate(metrics_names):
            axes[-1].bar(x + idx * width, metrics_values[metric_name], width, label=metric_name)

        axes[-1].set_xlabel('Детекторы')
        axes[-1].set_ylabel('Метрики')
        axes[-1].set_title('Сравнение метрик качества')
        axes[-1].set_xticks(x + width * 1.5)
        axes[-1].set_xticklabels(detector_names, rotation=45)
        axes[-1].legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _plot_online_results(X, predictions, scores):
        """Визуализация результатов онлайн-обучения"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Исходные данные
        axes[0].plot(X, 'b-', alpha=0.7)
        axes[0].set_title('Потоковые данные с дрейфом')
        axes[0].axvline(x=len(X) // 2, color='red', linestyle='--', alpha=0.7, label='Начало дрейфа')
        axes[0].legend()

        # Предсказания
        for idx, (name, pred) in enumerate(predictions.items()):
            anomaly_indices = np.where(pred == 1)[0]
            axes[1].scatter(anomaly_indices, np.ones(len(anomaly_indices)) * idx,
                            label=name, alpha=0.7)

        axes[1].set_yticks(range(len(predictions)))
        axes[1].set_yticklabels(list(predictions.keys()))
        axes[1].set_title('Обнаруженные аномалии')
        axes[1].set_xlabel('Время')
        axes[1].legend()

        # Оценки аномальности
        for name, score in scores.items():
            axes[2].plot(score, label=name, alpha=0.7)

        axes[2].set_title('Оценки аномальности')
        axes[2].set_xlabel('Время')
        axes[2].legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Запуск демонстраций
    univariate_results = AnomalyDetectionDemo.demo_univariate()
    multivariate_results = AnomalyDetectionDemo.demo_multivariate()
    online_predictions, online_scores = AnomalyDetectionDemo.demo_online_learning()
