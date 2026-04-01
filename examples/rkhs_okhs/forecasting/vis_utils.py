import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from okhs_forecasting_utils import (
    build_forecaster_params,
    extract_training_history,
)
from fedot_ind.core.models.kernel.okhs_forecasting_torch import OKHSForecasterTorch


class OKHSForecasterWithVisualization:

    def create_data(self):
        """Создание тестовых данных с различными паттернами"""
        np.random.seed(42)
        torch.manual_seed(42)

        # Создаем разнообразные временные ряды для тестирования
        self.time_points = np.linspace(0, 20, 200)

        # 1. Синусоидальный ряд с шумом
        self.sine_series = np.sin(self.time_points) + 0.1 * np.random.normal(size=len(self.time_points))

        # 2. Трендовый ряд
        self.trend_series = 0.1 * self.time_points + np.sin(self.time_points * 0.5) + 0.05 * np.random.normal(
            size=len(self.time_points))

        # 3. Сезонный ряд с множественными частотами
        self.seasonal_series = (
                np.sin(self.time_points) +
                0.5 * np.sin(2 * self.time_points) +
                0.2 * np.sin(4 * self.time_points) +
                0.1 * np.random.normal(size=len(self.time_points))
        )

        # 4. Нестационарный ряд с изменяющейся волатильностью
        self.volatile_series = np.sin(self.time_points) * (
                1 + 0.3 * np.sin(0.3 * self.time_points)) + 0.1 * np.random.normal(size=len(self.time_points))
        ts_monash = pd.read_csv('examples/rkhs_okhs/forecasting/MonashBitcoin_30.csv')
        self.monash_series = ts_monash[ts_monash['label'] == 'price']['value'].values
        self.series_dict = {
            'Синусоидальный': self.sine_series,
            'Трендовый': self.trend_series,
            'Сезонный': self.seasonal_series,
            'Волатильный': self.volatile_series,
            # 'Реальные данные':self.monash_series
        }

    def _prepare_forecast_for_viz(self, forecast, horizon, split_idx, time_series, ax, color):
        # Визуализация прогноза
        forecast_start = split_idx
        forecast_end = forecast_start + len(forecast)
        forecast_axis = np.arange(forecast_start, forecast_end)

        ax.plot(forecast_axis, forecast,
                color=color, linewidth=2, linestyle='-',
                marker='o', markersize=4,
                label=f'Прогноз (h={horizon})')

        # Заполнение области между прогнозом и реальными значениями
        if forecast_end <= len(time_series):
            real_values = time_series[forecast_start:forecast_end]
            ax.fill_between(forecast_axis, forecast, real_values,
                            color=color, alpha=0.2,
                            label=f'Ошибка (h={horizon})')
        return forecast_start, forecast_end, ax

    def _eval_metrics_on_horizon(self, horizon, time_series, forecast, forecast_start, forecast_end):
        # Вычисление метрик качества
        if forecast_end <= len(time_series):
            real_values = time_series[forecast_start:forecast_end]
            mae = np.mean(np.abs(forecast - real_values))
            mse = np.mean((forecast - real_values) ** 2)

            print(f"  Horizon {horizon}: MAE = {mae:.4f}, MSE = {mse:.4f}")

    def test_forecaster_with_visualization(self,
                                           horizons: list = [10],
                                           window_size: int = 40,
                                           model_params: dict = {}):
        """Тест с комплексной визуализацией прогнозов"""

        # Настройка стиля графиков
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Создаем фигуру для всех визуализаций
        fig = plt.figure(figsize=(20, 16))

        for idx, (series_name, time_series) in enumerate(self.series_dict.items()):
            print(f"\n=== Тестирование на ряде: {series_name} ===")

            # Тестируем разные горизонты прогнозирования
            colors = ['red', 'blue', 'green']

            # Разделяем на обучающую и тестовую выборки
            split_idx = int(len(time_series) * 0.8)
            train_series = time_series[:split_idx]
            test_series = time_series[split_idx:]

            # Создаем субплотов для каждого ряда
            ax = plt.subplot(2, 2, idx + 1)

            # Визуализация исходного ряда
            time_axis = np.arange(len(time_series))
            ax.plot(time_axis, time_series, 'k-', linewidth=2, alpha=0.7, label='Исходный ряд')
            ax.axvline(x=split_idx, color='gray', linestyle='--', alpha=0.7, label='Разделение train/test')

            # Прогнозы для разных горизонтов
            for horizon, color in zip(horizons, colors):
                # Создаем прогнозирующую модель с конкретным горизонтом
                horizon_model_params = build_forecaster_params(
                    model_params,
                    forecast_horizon=horizon,
                )
                horizon_forecaster = OKHSForecasterTorch(horizon_model_params)
                horizon_forecaster.fit(train_series, window_size=window_size)
                # Получаем прогноз
                forecast = horizon_forecaster.predict()
                if len(forecast) > 0:
                    # Визуализация прогноза
                    forecast_start, forecast_end, ax = self._prepare_forecast_for_viz(forecast,
                                                                                      horizon, split_idx, time_series,
                                                                                      ax, color)

                    # Вычисление метрик качества
                    self._eval_metrics_on_horizon(horizon, time_series, forecast, forecast_start, forecast_end)

            # Настройка графика
            ax.set_title(f'{series_name} ряд\nПрогнозы с разными горизонтами', fontsize=14, fontweight='bold')
            ax.set_xlabel('Время')
            ax.set_ylabel('Значение')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Добавляем информацию о модели
            model_info = horizon_forecaster.get_optimization_info()
            textstr = f"q={model_info['q']}\nУстройство: {model_info['device']}"
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.suptitle('Сравнение прогнозов OKHSForecaster для различных типов временных рядов',
                     fontsize=16, fontweight='bold', y=1.02)
        kernel_type = model_params['kernel_type']
        method_type = model_params['method']
        plt.savefig(f'Method - {method_type}.Kernel_type - {kernel_type}_okhs_forecaster_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.show()
        _ = 1

    def test_fractional_order_comparison(self):
        """Сравнение различных дробных порядков q"""

        # Используем синусоидальный ряд для сравнения
        time_series = self.sine_series
        split_idx = int(len(time_series) * 0.8)
        train_series = time_series[:split_idx]
        test_series = time_series[split_idx:]

        # Различные дробные порядки для сравнения
        q_values = [0.3, 0.5, 0.7, 0.9]
        colors = plt.cm.viridis(np.linspace(0, 1, len(q_values)))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # График 1: Визуализация прогнозов
        time_axis = np.arange(len(time_series))
        ax1.plot(time_axis, time_series, 'k-', linewidth=3, alpha=0.8, label='Исходный ряд')
        ax1.axvline(x=split_idx, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Разделение')

        metrics_data = []

        for q, color in zip(q_values, colors):
            print(f"\nТестирование с q={q}")

            forecaster = OKHSForecasterTorch(
                build_forecaster_params(
                    q=q,
                    forecast_horizon=20,
                    max_epochs=100,
                    device='cpu',
                    method='dmd',
                )
            )

            forecaster.fit(train_series, window_size=30)
            predictions = forecaster.predict()

            if len(predictions) > 0:
                forecast_start = split_idx
                forecast_end = forecast_start + len(predictions)
                forecast_axis = np.arange(forecast_start, forecast_end)

                ax1.plot(forecast_axis, predictions,
                         color=color, linewidth=2,
                         label=f'q={q}', alpha=0.8)

                # Вычисление метрик
                if forecast_end <= len(time_series):
                    real_values = time_series[forecast_start:forecast_end]
                    mae = np.mean(np.abs(predictions - real_values))
                    mse = np.mean((predictions - real_values) ** 2)

                    metrics_data.append({'q': q, 'MAE': mae, 'MSE': mse})
                    print(f"  q={q}: MAE = {mae:.4f}, MSE = {mse:.4f}")

        ax1.set_title('Сравнение прогнозов для различных дробных порядков q', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Время')
        ax1.set_ylabel('Значение')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # График 2: Метрики качества
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            x_pos = np.arange(len(metrics_df))

            bars1 = ax2.bar(x_pos - 0.2, metrics_df['MAE'], 0.4,
                            label='MAE', alpha=0.7, color='skyblue')
            bars2 = ax2.bar(x_pos + 0.2, metrics_df['MSE'], 0.4,
                            label='MSE', alpha=0.7, color='lightcoral')

            ax2.set_xlabel('Дробный порядок q')
            ax2.set_ylabel('Значение метрики')
            ax2.set_title('Метрики качества для различных q', fontsize=14, fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([f'{q:.1f}' for q in metrics_df['q']])
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Добавление значений на столбцы
            for bar, value in zip(bars1, metrics_df['MAE']):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                         f'{value:.4f}', ha='center', va='bottom', fontsize=9)

            for bar, value in zip(bars2, metrics_df['MSE']):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                         f'{value:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig('fractional_order_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def test_method_comparison(self):
        """Сравнение различных методов прогнозирования"""

        time_series = self.trend_series
        split_idx = int(len(time_series) * 0.8)
        train_series = time_series[:split_idx]

        methods = ['dmd', 'direct']
        method_names = ['DMD с функцией Миттаг-Леффлера', 'Прямой OKHS метод']
        colors = ['blue', 'red']

        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        axes = axes.flatten()

        time_axis = np.arange(len(time_series))

        for idx, (method, method_name, color) in enumerate(zip(methods, method_names, colors)):
            print(f"\nТестирование метода: {method_name}")

            # График прогнозов
            ax1 = axes[idx * 2]
            ax2 = axes[idx * 2 + 1]

            # Визуализация исходного ряда
            ax1.plot(time_axis, time_series, 'k-', linewidth=2, alpha=0.7, label='Исходный ряд')
            ax1.axvline(x=split_idx, color='gray', linestyle='--', alpha=0.7, label='Разделение')

            # Тестируем разные горизонты
            horizons = [10, 20]
            horizon_colors = ['lightcoral', 'lightgreen']

            for horizon, h_color in zip(horizons, horizon_colors):
                forecaster = OKHSForecasterTorch(
                    build_forecaster_params(
                        q=0.7,
                        forecast_horizon=horizon,
                        max_epochs=100,
                        device='cpu',
                        method=method,
                    )
                )

                forecaster.fit(train_series, window_size=30)
                predictions = forecaster.predict()

                if len(predictions) > 0:
                    forecast_start = split_idx
                    forecast_end = forecast_start + len(predictions)
                    forecast_axis = np.arange(forecast_start, forecast_end)

                    ax1.plot(forecast_axis, predictions,
                             color=h_color, linewidth=2, linestyle='-',
                             marker='s', markersize=3,
                             label=f'Прогноз h={horizon}')

                    # Визуализация ошибки
                    if forecast_end <= len(time_series):
                        real_values = time_series[forecast_start:forecast_end]
                        errors = np.abs(predictions - real_values)

                        ax2.bar(horizon, np.mean(errors),
                                color=h_color, alpha=0.7,
                                label=f'h={horizon}')

                        print(f"  Метод {method}, h={horizon}: Средняя ошибка = {np.mean(errors):.4f}")

            ax1.set_title(f'{method_name}\nПрогнозы с разными горизонтами', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Время')
            ax1.set_ylabel('Значение')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.set_title(f'{method_name}\nСредняя абсолютная ошибка', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Горизонт прогнозирования')
            ax2.set_ylabel('Средняя абсолютная ошибка')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle('Сравнение методов прогнозирования в OKHSForecaster',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def test_convergence_analysis(self):
        """Анализ сходимости обучения"""

        time_series = self.sine_series[:100]  # Используем укороченный ряд для скорости
        train_series = time_series[:80]

        # Тестируем различные параметры обучения
        learning_rates = [0.001, 0.01, 0.1]
        max_epochs_list = [50, 100, 200]

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

        convergence_data = []

        for idx, (lr, max_epochs) in enumerate(zip(learning_rates, max_epochs_list)):
            ax = axes[idx]

            print(f"\nАнализ сходимости: lr={lr}, max_epochs={max_epochs}")

            forecaster = OKHSForecasterTorch(
                build_forecaster_params(
                    q=0.7,
                    forecast_horizon=10,
                    max_epochs=max_epochs,
                    learning_rate=lr,
                    device='cpu',
                    method='dmd',
                )
            )

            forecaster.fit(train_series, window_size=20)
            losses = extract_training_history(forecaster)

            # Визуализация сходимости
            ax.plot(range(len(losses)), losses, 'b-', linewidth=2, alpha=0.8)
            ax.set_title(f'Сходимость (lr={lr}, epochs={max_epochs})', fontsize=12, fontweight='bold')
            ax.set_xlabel('Эпоха')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')  # Логарифмическая шкала для лучшей визуализации

            convergence_data.append({
                'learning_rate': lr,
                'max_epochs': max_epochs,
                'final_loss': losses[-1] if losses else float('inf'),
                'convergence_epoch': len(losses)
            })

        # Оставшиеся subplots используем для сводной информации
        if convergence_data:
            summary_ax = axes[-1]
            convergence_df = pd.DataFrame(convergence_data)

            # Визуализация финальных loss
            bars = summary_ax.bar(range(len(convergence_df)),
                                  convergence_df['final_loss'],
                                  color=plt.cm.viridis(np.linspace(0, 1, len(convergence_df))))

            summary_ax.set_title('Финальные значения loss', fontsize=12, fontweight='bold')
            summary_ax.set_xlabel('Конфигурация')
            summary_ax.set_ylabel('Финальный loss')
            summary_ax.set_xticks(range(len(convergence_df)))
            summary_ax.set_xticklabels([f'lr={row["learning_rate"]}\nep={row["max_epochs"]}'
                                        for _, row in convergence_df.iterrows()],
                                       rotation=45)
            summary_ax.grid(True, alpha=0.3)

            # Добавление значений на столбцы
            for bar, value in zip(bars, convergence_df['final_loss']):
                summary_ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                                f'{value:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.suptitle('Анализ сходимости обучения OKHSForecaster',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
