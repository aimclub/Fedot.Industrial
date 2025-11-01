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
    FORECASTING_PARAMS['method'] = 'dmd'
    experiment_handler.test_forecaster_with_visualization(model_params=FORECASTING_PARAMS)
    for kernel in KERNEL_TYPES:
        FORECASTING_PARAMS['kernel_type'] = kernel
        FORECASTING_PARAMS['method'] = 'occupation'
        experiment_handler.test_forecaster_with_visualization(model_params=FORECASTING_PARAMS)
    experiment_handler.test_method_comparison()
