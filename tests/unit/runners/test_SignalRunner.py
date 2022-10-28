from core.models.signal.SignalRunner import SignalRunner
from core.operation.utils.load_data import DataLoader


def extract_from_binary():
    signal_runner = SignalRunner()
    name = 'Earthquakes'
    train, test = DataLoader(name).load_data()
    train_feats = signal_runner.get_features(train[0], name)
    test_feats = signal_runner.get_features(test[0], name)
    return train_feats, test_feats


def extract_from_multi_class():
    signal_runner = SignalRunner()
    name = 'Lightning7'
    train, test = DataLoader(name).load_data()
    train_feats = signal_runner.get_features(train[0], name)
    test_feats = signal_runner.get_features(test[0], name)
    return train_feats, test_feats


def test_get_features():

    train_feats_earthquakes, test_feats_earthquakes = extract_from_binary()
    train_feats_lightning7, test_feats_lightning7 = extract_from_multi_class()
    assert train_feats_lightning7.shape == (70, 42)
    assert test_feats_lightning7.shape == (73, 43)

    assert train_feats_earthquakes.shape == (322, 43)
    assert test_feats_earthquakes.shape == (139, 43)
