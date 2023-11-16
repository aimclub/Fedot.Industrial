from fedot_ind.core.architecture.experiment.TimeSeriesAnomalyDetection import TimeSeriesAnomalyDetectionPreset
from fedot_ind.tools.synthetic.ts_generator import TimeSeriesGenerator

import pytest


@pytest.fixture()
def time_series():
    ts_config = {'ts_type': 'random_walk',
                 'length': 1000,
                 'start_val': 36.6}
    ts = TimeSeriesGenerator(ts_config).get_ts()
    return ts


@pytest.fixture()
def anomaly_dict():
    anomaly_d = {'anomaly1': [[40, 50], [60, 80], [200, 220]],
                 'anomaly2': [[300, 320], [400, 420], [600, 620]]}
    return anomaly_d


@pytest.fixture()
def detector():
    params = dict(branch_nodes=['eigen_basis'],
                  dataset='test',
                  tuning_iterations=1,
                  tuning_timeout=1,
                  model_params=dict(problem='classification',
                                    timeout=0.5,
                                    n_jobs=1,
                                    logging_level=50))
    detector = TimeSeriesAnomalyDetectionPreset(params)
    return detector


def test_fit_predict(detector, time_series, anomaly_dict):
    try:
        detector.fit(time_series, anomaly_dict)
    except Exception as ex:
        detector.fit(time_series, anomaly_dict)
    labels = detector.predict(time_series)
    proba = detector.predict_proba(time_series)
    metrics = detector.get_metrics(time_series, metric_names=['f1', 'roc_auc'])
    assert detector.auto_model.current_pipeline.is_fitted is True
