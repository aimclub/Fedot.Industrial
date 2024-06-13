import unittest
import numpy as np
from fedot.core.data.data import InputData
from fedot_ind.core.models.ts_forecasting.ssa_forecaster import SSAForecasterImplementation

class TestSSAForecasterImplementation(unittest.TestCase):
    def test_default_initialization(self):
        forecaster = SSAForecasterImplementation()
        self.assertIsInstance(forecaster, SSAForecasterImplementation)
        self.assertEqual(forecaster.window_size_method, None)
        self.assertEqual(forecaster.history_lookback, 100)
        self.assertEqual(forecaster.low_rank_approximation, False)
        self.assertEqual(forecaster.tuning_params, {'tuning_iterations': 100, 'tuning_timeout': 20, 'tuning_early_stop': 20, 'tuner': 'SimultaneousTuner'})
        self.assertEqual(forecaster.component_model, 'topological')
        self.assertEqual(forecaster.mode, 'channel_independent')

    def test_custom_initialization(self):
        params = {
            'window_size_method': 'hac',
            'history_lookback': 50,
            'low_rank_approximation': True,
            'tuning_params': {'tuning_iterations': 50, 'tuning_timeout': 10, 'tuning_early_stop': 10, 'tuner': 'OptunaTuner'},
            'component_model': 'ar',
            'mode': 'one_dimensional'
        }
        forecaster = SSAForecasterImplementation(params)
        self.assertIsInstance(forecaster, SSAForecasterImplementation)
        self.assertEqual(forecaster.window_size_method, 'hac')
        self.assertEqual(forecaster.history_lookback, 50)
        self.assertEqual(forecaster.low_rank_approximation, True)
        self.assertEqual(forecaster.tuning_params, {'tuning_iterations': 50, 'tuning_timeout': 10, 'tuning_early_stop': 10, 'tuner': 'OptunaTuner'})
        self.assertEqual(forecaster.component_model, 'ar')
        self.assertEqual(forecaster.mode, 'one_dimensional')

    def test_predict_one_dimensional(self):
        forecaster = SSAForecasterImplementation({'mode': 'one_dimensional'})
        time_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        input_data = InputData(features=time_series, target=time_series)
        forecast = forecaster.predict(input_data)
        self.assertIsInstance(forecast.predict, np.ndarray)
        self.assertEqual(forecast.predict.shape, (forecaster.horizon,))

    def test_predict_channel_independent(self):
        forecaster = SSAForecasterImplementation({'mode': 'channel_independent'})
        time_series = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        input_data = InputData(features=time_series, target=time_series)
        forecast = forecaster.predict(input_data)
        self.assertIsInstance(forecast.predict, np.ndarray)
        self.assertEqual(forecast.predict.shape, (forecaster.horizon,))

    def test_predict_missing_values(self):
        forecaster = SSAForecasterImplementation()
        time_series = np.array([1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10])
        input_data = InputData(features=time_series, target=time_series)
        forecast = forecaster.predict(input_data)
        self.assertIsInstance(forecast.predict, np.ndarray)
        self.assertEqual(forecast.predict.shape, (forecaster.horizon,))

    def test_predict_invalid_input(self):
        forecaster = SSAForecasterImplementation()
        input_data = InputData(features=None, target=None)
        with self.assertRaises(ValueError):
            forecaster.predict(input_data)

        input_data = InputData(features=np.array([]), target=np.array([]))
        with self.assertRaises(ValueError):
            forecaster.predict(input_data)

        input_data = InputData(features=np.array([1, 2, 3]), target=np.array([1, 2]))
        with self.assertRaises(ValueError):
            forecaster.predict(input_data)

if __name__ == '__main__':
    unittest.main()