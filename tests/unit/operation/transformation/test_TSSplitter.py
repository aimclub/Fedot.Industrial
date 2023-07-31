import unittest

import numpy as np

from fedot_ind.core.operation.transformation.splitter import TSSplitter


class TestAnomalyDetector(unittest.TestCase):
    def test_get_anomaly_intervals(self):
        data = np.random.rand(320)
        anomaly_dict = {'anomaly1': '40:50, 60:80',
                        'anomaly2': '130:170, 300:320'}
        splitter = TSSplitter(time_series=data, anomaly_dict=anomaly_dict)
        classes, anomaly_intervals = splitter._get_anomaly_intervals()

        expected_intervals = [[[40, 50],[60, 80]],
                              [[130, 170],[300, 320]]
                             ]

        self.assertEqual(anomaly_intervals, expected_intervals)
