import shutil

import numpy as np
import pytest

from examples.anomaly_detection.detection_classification import generate_time_series, split_series, \
    convert_anomalies_dict_to_points
from fedot_ind.api.main import FedotIndustrial


@pytest.mark.parametrize('dimension', [1, 3])
def test_anomaly_detection(dimension):
    np.random.seed(42)
    time_series, anomaly_intervals = generate_time_series(
        ts_length=1000,
        dimension=dimension,
        num_anomaly_classes=2,
        num_of_anomalies=50)

    series_train, anomaly_train, series_test, anomaly_test = split_series(time_series, anomaly_intervals, test_part=300)

    point_test = convert_anomalies_dict_to_points(series_test, anomaly_test)

    industrial = FedotIndustrial(task='anomaly_detection',
                                 dataset='custom_dataset',
                                 strategy='fedot_preset',
                                 use_cache=False,
                                 timeout=0.5,
                                 n_jobs=1,
                                 logging_level=20,
                                 output_folder='.')

    model = industrial.fit(features=series_train,
                           anomaly_dict=anomaly_train)

    industrial.solver.save('model')

    # prediction before loading
    labels_before = industrial.predict(features=series_test)
    probs_before = industrial.predict_proba(features=series_test)

    industrial.solver.load('model')

    # prediction after loading
    labels_after = industrial.predict(features=series_test)
    probs_after = industrial.predict_proba(features=series_test)

    metrics = industrial.solver.get_metrics(target=point_test,
                                            metric_names=['f1', 'roc_auc'])

    shutil.rmtree('model')

    assert np.all(labels_after == labels_before)

    assert metrics['f1'] > 0.5
    assert metrics['roc_auc'] > 0.5
