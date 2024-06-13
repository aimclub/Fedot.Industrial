import unittest
import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot_ind.core.models.ts_forecasting.ssa_forecaster import SSAForecasterImplementation


class TestSSAForecasterImplementation(unittest.TestCase):
    def test_init_default_params(self):
        forecaster = SSAForecasterImplementation()
        self.assertEqual(forecaster.window_size_method, None)
        self.assertEqual(forecaster.history_lookback, 100)
        self.assertEqual(forecaster.low_rank_approximation, False)
        self.assertEqual(forecaster.tuning_params,
                         {'tuning_iterations': 100,
                          'tuning_timeout': 20,
                          'tuning_early_stop': 20,
                          'tuner': 'SimultaneousTuner'})
        self.assertEqual(
            forecaster.component_model.root_node.operation.operation_type,
            'lagged')
        self.assertEqual(forecaster.mode, 'channel_independent')
        self.assertEqual(
            forecaster.trend_model.root_node.operation.operation_type,
            'lagged')
        self.assertIsNone(forecaster._decomposer)
        self.assertIsNone(forecaster._rank_thr)
        self.assertIsNone(forecaster._window_size)
        self.assertIsNone(forecaster.horizon)
        self.assertFalse(forecaster.preprocess_to_lagged)

    def test_init_custom_params(self):
        custom_params = {
            'window_size_method': 'hac',
            'history_lookback': 50,
            'low_rank_approximation': True,
            'tuning_params': {
                'tuning_iterations': 50,
                'tuning_timeout': 10,
                'tuning_early_stop': 10,
                'tuner': 'OptunaTuner'},
            'component_model': 'ar',
            'mode': 'one_dimensional'}
        forecaster = SSAForecasterImplementation(custom_params)
        self.assertEqual(forecaster.window_size_method, 'hac')
        self.assertEqual(forecaster.history_lookback, 50)
        self.assertTrue(forecaster.low_rank_approximation)
        self.assertEqual(forecaster.tuning_params,
                         {'tuning_iterations': 50,
                          'tuning_timeout': 10,
                          'tuning_early_stop': 10,
                          'tuner': 'OptunaTuner'})
        self.assertEqual(
            forecaster.component_model.root_node.operation.operation_type, 'ar')
        self.assertEqual(forecaster.mode, 'one_dimensional')
        self.assertEqual(
            forecaster.trend_model.root_node.operation.operation_type, 'ar')

    def test_predict_simple_ts(self):
        time_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        input_data = InputData(
            idx=np.arange(10),
            features=time_series,
            target=time_series,
            task=None)
        forecaster = SSAForecasterImplementation()
        forecaster.horizon = 3
        output_data = forecaster.predict(input_data)
        self.assertIsInstance(output_data, OutputData)
        self.assertEqual(output_data.predict.shape, (3,))

    def test_predict_complex_ts(self):
        time_series = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        input_data = InputData(
            idx=np.arange(4),
            features=time_series,
            target=time_series,
            task=None)
        forecaster = SSAForecasterImplementation()
        forecaster.horizon = 2
        output_data = forecaster.predict(input_data)
        self.assertIsInstance(output_data, OutputData)
        self.assertEqual(output_data.predict.shape, (2, 3))

    def test_predict_edge_cases(self):
        empty_ts = np.array([])
        empty_input_data = InputData(
            idx=np.array(
                []),
            features=empty_ts,
            target=empty_ts,
            task=None)
        forecaster = SSAForecasterImplementation()
        forecaster.horizon = 3
        with self.assertRaises(ValueError):
            forecaster.predict(empty_input_data)

        missing_values_ts = np.array([1, 2, np.nan, 4, 5])
        missing_values_input_data = InputData(
            idx=np.arange(5),
            features=missing_values_ts,
            target=missing_values_ts,
            task=None)
        output_data = forecaster.predict(missing_values_input_data)
        self.assertIsInstance(output_data, OutputData)
        self.assertEqual(output_data.predict.shape, (3,))

    def test_fit(self):
        time_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        input_data = InputData(
            idx=np.arange(10),
            features=time_series,
            target=time_series,
            task=None)
        forecaster = SSAForecasterImplementation()
        forecaster.fit(input_data)

    def test_predict_for_fit_simple_ts(self):
        time_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        input_data = InputData(
            idx=np.arange(10),
            features=time_series,
            target=time_series,
            task=None)
        forecaster = SSAForecasterImplementation()
        forecaster.horizon = 3
        predicted_values = forecaster.predict_for_fit(input_data)
        self.assertEqual(predicted_values.shape, (10,))

    def test_predict_for_fit_complex_ts(self):
        time_series = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        input_data = InputData(
            idx=np.arange(4),
            features=time_series,
            target=time_series,
            task=None)
        forecaster = SSAForecasterImplementation()
        forecaster.horizon = 2
        predicted_values = forecaster.predict_for_fit(input_data)
        self.assertEqual(predicted_values.shape, (4, 3))

    def test_predict_for_fit_edge_cases(self):
        empty_ts = np.array([])
        empty_input_data = InputData(
            idx=np.array(
                []),
            features=empty_ts,
            target=empty_ts,
            task=None)
        forecaster = SSAForecasterImplementation()
        forecaster.horizon = 3
        with self.assertRaises(ValueError):
            forecaster.predict_for_fit(empty_input_data)

        missing_values_ts = np.array([1, 2, np.nan, 4, 5])
        missing_values_input_data = InputData(
            idx=np.arange(5),
            features=missing_values_ts,
            target=missing_values_ts,
            task=None)
        predicted_values = forecaster.predict_for_fit(
            missing_values_input_data)
        self.assertEqual(predicted_values.shape, (5,))


if __name__ == '__main__':
    unittest.main()
