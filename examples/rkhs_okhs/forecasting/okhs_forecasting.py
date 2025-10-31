import matplotlib.pyplot as plt
import numpy as np

from examples.rkhs_okhs.forecasting.vis_utils import OKHSForecasterWithVisualization
from fedot_ind.core.models.kernel.okhs_forecasting_torch import OKHSForecasterTorch

FORECASTING_PARAMS = dict(q=0.3, forecast_horizon=20, epochs=3000)


def run_test(window_size: int = 20):
    """Быстрый визуальный тест для отладки"""
    # Простой тестовый ряд
    t = np.linspace(0, 10, 100)
    series = np.sin(t) + 0.1 * np.random.normal(size=100)

    forecaster = OKHSForecasterTorch(FORECASTING_PARAMS)

    forecaster.fit(series, window_size=window_size)
    predictions = forecaster.predict()

    plt.figure(figsize=(12, 6))
    plt.plot(series, 'b-', label='Исходный ряд', linewidth=2)
    plt.plot(range(len(series) - len(predictions), len(series)),
             predictions, 'r-', label='Прогноз', linewidth=2, marker='o')
    plt.axvline(x=len(series) - len(predictions), color='gray', linestyle='--', label='Начало прогноза')
    plt.title('Быстрый тест OKHSForecaster')
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return predictions


if __name__ == "__main__":
    # # Дополнительный быстрый тест
    # print("\n" + "=" * 60)
    # print("БЫСТРЫЙ ВИЗУАЛЬНЫЙ ТЕСТ")
    # print("=" * 60)
    # for ws in [10,20,40]:
    #     predictions = run_test(ws)
    # _ = 1
    experiment_handler = OKHSForecasterWithVisualization()
    experiment_handler.create_data()
    experiment_handler.test_forecaster_with_visualization(model_params=FORECASTING_PARAMS)
    experiment_handler.test_method_comparison()
