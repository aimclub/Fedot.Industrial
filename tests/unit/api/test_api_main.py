import pytest
from unittest.mock import MagicMock
import pandas as pd
from fedot_ind.api.main import FedotIndustrial





@pytest.fixture
def fedot_industrial_regression():
    return FedotIndustrial(problem='regression', output_folder='test_output')




def test_predict_classification(fedot_industrial, ):
    train_data = (pd.DataFrame(), pd.Series())
    pipeline = fedot_industrial.fit(train_data)
    assert isinstance(pipeline, FedotIndustrial)

def test_predict_proba_classification(fedot_industrial, ):
    train_data = (pd.DataFrame(), pd.Series())
    pipeline = fedot_industrial.fit(train_data)
    assert isinstance(pipeline, FedotIndustrial)


def test_predict(fedot_industrial):
    predict_data = pd.DataFrame()
    fedot_industrial.solver = MagicMock()
    labels = fedot_industrial.predict(predict_data)
    assert labels is not None


def test_predict_proba(fedot_industrial):
    predict_data = pd.DataFrame()
    fedot_industrial.solver = MagicMock()
    probs = fedot_industrial.predict_proba(predict_data)
    assert probs is not None


def test_finetune(fedot_industrial):
    train_data = (pd.DataFrame(), pd.Series())
    fedot_industrial.solver = MagicMock()
    fedot_industrial.finetune(train_data)
    assert fedot_industrial.solver.fit.called


def test_get_metrics(fedot_industrial):
    target = pd.Series()
    metrics = fedot_industrial.get_metrics(target=target, metric_names=['f1'])
    assert isinstance(metrics, pd.DataFrame)


def test_save_predict(fedot_industrial):
    predicted_data = pd.Series()
    fedot_industrial.solver = MagicMock()
    fedot_industrial.save_predict(predicted_data)
    assert fedot_industrial.solver.save_prediction.called


def test_save_metrics(fedot_industrial):
    metrics = {'f1': 0.8}
    fedot_industrial.solver = MagicMock()
    fedot_industrial.save_metrics(metrics=metrics)
    assert fedot_industrial.solver.save_metrics.called


def test_load(fedot_industrial):
    model_path = 'test_model_path'
    fedot_industrial.current_pipeline = MagicMock()
    fedot_industrial.load(model_path)
    assert fedot_industrial.current_pipeline.load.called


def test_save_optimization_history(fedot_industrial):
    fedot_industrial.solver = MagicMock()
    fedot_industrial.save_optimization_history()
    assert fedot_industrial.solver.history.save.called


def test_save_best_model(fedot_industrial):
    fedot_industrial.solver = MagicMock()
    fedot_industrial.save_best_model()
    assert fedot_industrial.solver.current_pipeline.show.called


def test_plot_fitness_by_generation(fedot_industrial):
    fedot_industrial.solver = MagicMock()
    fedot_industrial.plot_fitness_by_generation()
    assert fedot_industrial.solver.history.show.fitness_box.called


def test_plot_operation_distribution(fedot_industrial):
    fedot_industrial.solver = MagicMock()
    fedot_industrial.plot_operation_distribution()
    assert fedot_industrial.solver.history.show.operations_kde.called


def test_explain(fedot_industrial):
    fedot_industrial.solver = MagicMock()
    fedot_industrial.predict_data = pd.DataFrame()
    fedot_industrial.explain()
    assert fedot_industrial.solver.explain.called


def test_generate_ts(fedot_industrial):
    ts_config = {'length': 100, 'trend': 0.01}
    ts_data = fedot_industrial.generate_ts(ts_config)
    assert ts_data is not None


def test_generate_anomaly_ts(fedot_industrial):
    ts_data = pd.Series()
    anomaly_config = {'magnitude': 0.1, 'start': 20, 'end': 30}
    init_ts, mod_ts, synth_inters = fedot_industrial.generate_anomaly_ts(ts_data, anomaly_config)
    assert init_ts is not None
    assert mod_ts is not None
    assert synth_inters is not None


def test_split_ts(fedot_industrial):
    ts_data = pd.Series()
    anomaly_dict = {1: (10, 20), 2: (30, 40)}
    features, target = fedot_industrial.split_ts(ts_data, anomaly_dict)
    assert isinstance(features, pd.DataFrame)
    assert isinstance(target, pd.Series)
