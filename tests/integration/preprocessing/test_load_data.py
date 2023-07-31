from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader


def test_load_multivariate_data():
    train_data, test_data = DataLoader('Epilepsy').load_data()
    x_train, y_train = train_data
    x_test, y_test = test_data
    assert x_train.shape == (137, 3)
    assert x_test.shape == (138, 3)
    assert y_train.shape == (137,)
    assert y_test.shape == (138,)


def test_load_univariate_data():
    train_data, test_data = DataLoader('DodgerLoopDay').load_data()
    x_train, y_train = train_data
    x_test, y_test = test_data
    assert x_train.shape == (78, 288)
    assert x_test.shape == (80, 288)
    assert y_train.shape == (78,)
    assert y_test.shape == (80,)


def test_load_fake_data():
    train_data, test_data = DataLoader('Fake').load_data()
    assert train_data is None
    assert test_data is None

