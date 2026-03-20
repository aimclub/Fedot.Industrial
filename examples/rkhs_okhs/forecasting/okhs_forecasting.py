from examples.rkhs_okhs.forecasting.okhs_forecasting_utils import build_forecaster_params
from examples.rkhs_okhs.forecasting.vis_utils import OKHSForecasterWithVisualization

# BAD KERNELS - 'rational_quadratic', 'graph_diffusion','spectral_mixture','adaptive'

KERNEL_TYPES = [
    'rbf',
    'periodic',
    'fractional',
    'linear',
    'polynomial',
    'matern',
]
MODEL_BACKEND = ['occupation', 'dmd']
FORECASTING_PARAMS = dict(q=0.5, kernel_type='periodic', forecast_horizon=20, epochs=500)

if __name__ == "__main__":
    experiment_handler = OKHSForecasterWithVisualization()
    experiment_handler.create_data()
    dmd_params = build_forecaster_params(FORECASTING_PARAMS, method='dmd')
    experiment_handler.test_forecaster_with_visualization(model_params=dmd_params)
    for kernel in KERNEL_TYPES:
        occupation_params = build_forecaster_params(
            FORECASTING_PARAMS,
            kernel_type=kernel,
            method='occupation',
        )
        experiment_handler.test_forecaster_with_visualization(model_params=occupation_params)
    experiment_handler.test_method_comparison()
