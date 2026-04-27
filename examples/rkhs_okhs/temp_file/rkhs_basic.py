import numpy as np

from examples.rkhs_okhs.forecasting.okhs_forecasting_utils import build_forecaster_params
from fedot_ind.core.models.kernel.okhs_forecasting_torch import OKHSForecasterTorch
from fedot_ind.core.operation.transformation.representation.kernel.kernels import OccupationKernel


def create_trajectory():
    trajectories = [
        np.sin(np.linspace(0, 2 * np.pi, 20)) + 0.1 * np.random.normal(size=20),
        np.cos(np.linspace(0, 2 * np.pi, 20)) + 0.1 * np.random.normal(size=20),
        np.sin(np.linspace(0, 2 * np.pi, 20) + 0.5) + 0.1 * np.random.normal(size=20),
    ]
    return trajectories


def create_lagged_ts():
    t = np.linspace(0, 10, 100)
    data = np.sin(t) + 0.3 * np.sin(3 * t) + 0.1 * np.random.normal(size=100)
    return data, [data[i:i + 20] for i in range(0, 80, 5)]


def demo_occupation_kernel():
    print("=== Демонстрация Occupation Kernel ===")

    trajectories = create_trajectory()
    kernel = OccupationKernel(q=0.7)
    gram_matrix = kernel.compute_gram_matrix(trajectories)

    print("Матрица Грама:")
    print(gram_matrix)


def demo_okhs_forecasting():
    print("\n=== Демонстрация OKHS forecasting ===")

    time_series, _ = create_lagged_ts()
    forecaster = OKHSForecasterTorch(
        build_forecaster_params(
            q=0.8,
            forecast_horizon=10,
            epochs=20,
            device="cpu",
            method="occupation",
            forecasting_strategy="recursive",
        )
    )
    forecaster.fit(time_series, window_size=20)
    prediction = forecaster.predict()

    print(f"Прогноз на {len(prediction)} шагов:")
    print(prediction.flatten())


if __name__ == "__main__":
    demo_occupation_kernel()
    demo_okhs_forecasting()
