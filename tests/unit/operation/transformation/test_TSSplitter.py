import unittest

import numpy as np
import pytest

from fedot_ind.core.operation.transformation.splitter import TSSplitter


# 1 case: аномалии стоят по краям - проверить корректность новых границ
# 2 case: аномалии стоят в середине - проверить корректность границ
# 3 case: аномалии стоят в середине и по краям - проверить корректность границ
# 4 case: аномалии слишком большие - должна выпасть ошибка, что нет неаномальных сэмплов
# 5 case:


class TestAnomalyDetector(unittest.TestCase):
    def test_get_anomaly_intervals(self):
        data = np.random.rand(320)
        anomaly_dict = {'anomaly1': [[40, 50], [60, 80]],
                        'anomaly2': [[130, 170], [300, 320]]}
        splitter = TSSplitter(time_series=data, anomaly_dict=anomaly_dict)
        classes, anomaly_intervals = splitter._get_anomaly_intervals()

        expected_intervals = [[[40, 50], [60, 80]],
                              [[130, 170], [300, 320]]]

        self.assertEqual(anomaly_intervals, expected_intervals)

    def test_get_frequent_anomaly_length(self):
        # Test case with no frequent anomalies
        data = [0, 0, 0, 0, 0, 0, 0]
        anomaly_intervals = [(1, 2), (4, 6)]
        frequent_length = self.detector._get_frequent_anomaly_length(anomaly_intervals, data)
        self.assertEqual(frequent_length, 0)

        # Test case with one frequent anomaly
        data = [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
        anomaly_intervals = [(1, 3), (5, 8), (10, 12)]
        frequent_length = self.detector._get_frequent_anomaly_length(anomaly_intervals, data)
        self.assertEqual(frequent_length, 3)

        # Test case with multiple frequent anomalies of same length
        data = [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0]
        anomaly_intervals = [(1, 4), (7, 11)]
        frequent_length = self.detector._get_frequent_anomaly_length(anomaly_intervals, data)
        self.assertEqual(frequent_length, 4)

        # Test case with multiple frequent anomalies of different lengths
        data = [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
        anomaly_intervals = [(1, 3), (5, 7), (10, 12)]
        frequent_length = self.detector._get_frequent_anomaly_length(anomaly_intervals, data)
        self.assertEqual(frequent_length, 2)


@pytest.fixture()
def splitter():
    pass
