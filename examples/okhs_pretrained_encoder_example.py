"""
Пример использования предобученного DeepFDMDAutoencoder ядра с OKHS fDMD.

Этот пример показывает, как встроить обученный энкодер в OKHS pipeline для улучшения качества предсказаний.
"""

import numpy as np
import torch
from pathlib import Path

from fedot_ind.core.models.ts_forecasting.dmd_models.okhs_fdmd_forecaster import (
    OKHSFDMDForecaster,
    build_okhs_fdmd_forecaster,
)
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.deep_fdmd_net import (
    DeepFDMDAutoencoder,
)


def example_1_with_saved_weights():
    """Пример 1: Использование энкодера из сохраненных весов."""
    print("=" * 60)
    print("Пример 1: Загрузка энкодера из сохраненных весов")
    print("=" * 60)
    
    # Синтетический временной ряд
    time_series = np.cumsum(np.random.randn(100)) + 10
    
    # Параметры
    params = {
        'pretrained_encoder_path': 'saved_models/deep_autoencoder_weights.pth',
        'encoder_latent_dim': 16,  # должен совпадать с размером латентного пространства
        'q': 0.7,
        'n_modes': 5,
    }
    
    # Создание и обучение модели
    forecaster = OKHSFDMDForecaster(
        forecast_horizon=10,
        **params,
    )
    
    try:
        forecaster.fit(time_series)
        forecast = forecaster.predict()
        print(f"✓ Успешно! Предсказание: {forecast[:5]}...")
        print(f"  Диагностика: {forecaster.get_diagnostics()}")
    except FileNotFoundError:
        print(f"✗ Файл весов не найден. Создайте его используя примеры из deep_okhs_test.py")


def example_2_with_encoder_object():
    """Пример 2: Прямое использование объекта энкодера."""
    print("\n" + "=" * 60)
    print("Пример 2: Использование объекта энкодера напрямую")
    print("=" * 60)
    
    # Синтетический временной ряд
    np.random.seed(42)
    time_series = np.sin(np.linspace(0, 4 * np.pi, 200)) + np.random.randn(200) * 0.1
    
    # Создаем и "обучаем" энкодер (в примере просто создаем)
    input_dim = 1  # размерность входного пространства (1 для univariate TS)
    latent_dim = 8  # размер латентного пространства
    
    encoder = DeepFDMDAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    encoder.eval()  # режим оценки
    
    print(f"Энкодер создан: input_dim={input_dim}, latent_dim={latent_dim}")
    
    # Создание и обучение модели с предобученным энкодером
    forecaster = OKHSFDMDForecaster(
        forecast_horizon=20,
        q=0.7,
        n_modes=5,
        window_size=30,
        n_modes=5,
        pretrained_encoder=encoder,  # ← Передаем энкодер напрямую
        encoder_latent_dim=latent_dim,
    )
    
    print(f"OKHS fDMD Forecaster создан с предобученным энкодером")
    
    forecaster.fit(time_series)
    forecast = forecaster.predict()
    
    print(f"✓ Обучение завершено!")
    print(f"  Предсказание (первые 5 значений): {forecast[:5]}")
    print(f"  Горизонт: {len(forecast)}")
    
    # Получить диагностику
    diag = forecaster.get_diagnostics()
    print(f"  Параметр q (дробный порядок): {diag.get('resolved_q')}")


def example_3_comparison():
    """Пример 3: Сравнение с дефолтным ядром (без энкодера)."""
    print("\n" + "=" * 60)
    print("Пример 3: Сравнение - с энкодером vs без")
    print("=" * 60)
    
    # Синтетический временной ряд
    np.random.seed(42)
    t = np.linspace(0, 10, 300)
    time_series = np.sin(0.5 * t) * np.exp(-t / 20) + np.random.randn(len(t)) * 0.05
    
    forecast_horizon = 30
    
    # 1. Дефолтная модель (RBF kernel)
    print("\n1. Обучение модели без энкодера (RBF kernel)...")
    forecaster_default = OKHSFDMDForecaster(
        forecast_horizon=forecast_horizon,
        q=0.7,
        n_modes=5,
        window_size=40,
    )
    forecaster_default.fit(time_series)
    forecast_default = forecaster_default.predict()
    
    # 2. Модель с энкодером
    print("2. Обучение модели с энкодером (DeepKernel)...")
    encoder = DeepFDMDAutoencoder(input_dim=1, latent_dim=12)
    encoder.eval()
    
    forecaster_deep = OKHSFDMDForecaster(
        forecast_horizon=forecast_horizon,
        q=0.7,
        n_modes=5,
        window_size=40,
        pretrained_encoder=encoder,
        encoder_latent_dim=12,
    )
    forecaster_deep.fit(time_series)
    forecast_deep = forecaster_deep.predict()
    
    # Сравнение
    print(f"\nСравнение предсказаний:")
    print(f"{'Метрика':<20} | {'RBF Kernel':<15} | {'Deep Kernel':<15}")
    print("-" * 52)
    
    mse_default = np.mean((forecast_default - forecast_deep) ** 2)
    mae_default = np.mean(np.abs(forecast_default - forecast_deep))
    
    print(f"{'Первое значение':<20} | {forecast_default[0]:<15.4f} | {forecast_deep[0]:<15.4f}")
    print(f"{'Среднее значение':<20} | {np.mean(forecast_default):<15.4f} | {np.mean(forecast_deep):<15.4f}")
    print(f"{'Макс значение':<20} | {np.max(forecast_default):<15.4f} | {np.max(forecast_deep):<15.4f}")
    print(f"{'Разница MSE':<20} | {mse_default:<15.6f}")
    print(f"{'Разница MAE':<20} | {mae_default:<15.6f}")


def example_4_stage_tuning_with_encoder():
    """Пример 4: Использование encoder с stage tuning."""
    print("\n" + "=" * 60)
    print("Пример 4: Stage Tuning с предобученным энкодером")
    print("=" * 60)
    
    # Синтетический временной ряд
    np.random.seed(42)
    time_series = np.cumsum(np.random.randn(150)) + 50
    
    # Создание модели с энкодером
    encoder = DeepFDMDAutoencoder(input_dim=1, latent_dim=10)
    encoder.eval()
    
    model_implementation = OKHSFDMDForecaster(
        forecast_horizon=15,
        q=0.7,
        n_modes=4,
        window_size=25,
        pretrained_encoder=encoder,
        encoder_latent_dim=10,
    )
    
    print("OKHS fDMD с энкодером создан")
    print("Параметры можно дальше оптимизировать через stage_tuning mechanism")


if __name__ == "__main__":
    print("\n" + "╔" + "=" * 58 + "╗")
    print("║" + " OKHS fDMD с предобученным DeepFDMD энкодером".center(58) + "║")
    print("╚" + "=" * 58 + "╝" + "\n")
    
    # Запустить примеры
    example_2_with_encoder_object()  # Этот работает без сохраненных весов
    example_3_comparison()  # Сравнение
    
    # Эти требуют сохраненных весов:
    print("\n" + "=" * 60)
    print("Дополнительные примеры (требуют сохраненные веса):")
    print("=" * 60)
    example_1_with_saved_weights()
    example_4_stage_tuning_with_encoder()
    
    print("\n✓ Примеры завершены!")
