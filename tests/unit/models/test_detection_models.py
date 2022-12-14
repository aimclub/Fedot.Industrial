from statistics import mean
import numpy as np
import pytest
import sys
import os
import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(import_path)
sys.path.append(os.path.join(import_path, "../../../"))
from core.models.detection.subspaces.SSTdetector import SingularSpectrumTransformation
from core.models.detection.abstract_objects.FileObject import FileObject
from core.models.detection.area.ThresholdZonesDetector import ThresholdZonesDetector
from core.models.detection.vector.AngleBasedDetector import AngleBasedDetector

@pytest.fixture()
def basic_periodic_data():
    x0 = 1 * np.ones(1000) + np.random.rand(1000) * 1
    x1 = 3 * np.ones(1000) + np.random.rand(1000) * 2
    x2 = 5 * np.ones(1000) + np.random.rand(1000) * 1.5
    x = np.hstack([x0, x1, x2])
    x += np.random.rand(x.size)

    x0 = 6 * np.ones(1000) + np.random.rand(1000) * 1
    x1 = 3 * np.ones(1000) + np.random.rand(1000) * 2
    x2 = 7 * np.ones(1000) + np.random.rand(1000) * 1.5
    y = np.hstack([x0, x1, x2])
    y += np.random.rand(x.size)
    return x, y


def test_SST_detector(basic_periodic_data):
    ts_1, ts_2 = basic_periodic_data
    scorer = SingularSpectrumTransformation(time_series=ts_1,
                                            ts_window_length=100,
                                            lag=10,
                                            trajectory_window_length=30)
    score = scorer.score_offline(dynamic_mode=False)

def test_Threshold_Zones_Detector(basic_periodic_data):
    ts_1, ts_2 = basic_periodic_data
    file_object = FileObject(ts_1, "test")

    detector = ThresholdZonesDetector(0.5)
    detector.load_data(file_object)
    detector.run_operation()
    new_object: FileObject = detector.return_new_data()

    assert len(new_object.anomalies_list) is not 0
    assert new_object.anomalies_list[0].get_end() - new_object.anomalies_list[0].get_start() is not 0

def test_Angle_Based_Detector(basic_periodic_data):
    ts_1, ts_2 = basic_periodic_data
    data_dict = {
        "Ts_1": ts_1,
        "Ts_2": ts_2
    }
    file_object = FileObject(ts_1, "test")
    file_object.time_series_data = data_dict
    detector = AngleBasedDetector(300)
    detector.load_data(file_object)
    detector.run_operation()
    new_object: FileObject = detector.return_new_data()
    new_object.test_vector_ts
    assert len(new_object.test_vector_ts) is not 0
    assert mean(new_object.test_vector_ts) is not 0
