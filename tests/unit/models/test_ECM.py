import pandas as pd

from core.models.ecm.error_correction import Booster
import pytest


@pytest.fixture()
def loaded_predict_data():
    base_predict = pd.read_csv('base_predict.csv')
    return base_predict.values


@pytest.fixture()
def loaded_features_target():
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    return X_train.values, y_train.values


def test_booster(loaded_predict_data, loaded_features_target):
    booster = Booster(features_train=loaded_features_target[0],
                      target_train=loaded_features_target[1],
                      base_predict=loaded_predict_data,
                      timeout=1)

    result = booster.fit()
