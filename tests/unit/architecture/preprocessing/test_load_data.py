from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader


def test_load_data_dodger_loop_day():
    train_data, test_data = DataLoader('DodgerLoopDay').load_data()
    x_train, y_train = train_data
    x_test, y_test = test_data
    assert x_train.shape == (78, 288)
    assert x_test.shape == (80, 288)
    assert y_train.shape == (78,)
    assert y_test.shape == (80,)


def test_load_data_beef():
    train_data, test_data = DataLoader('Beef').load_data()
    assert train_data[0].shape == (30, 470)
    assert test_data[0].shape == (30, 470)
    assert train_data[1].shape == (30,)
    assert test_data[1].shape == (30,)


def test_load_data_coffee():
    train_data, test_data = DataLoader('Coffee').load_data()
    assert train_data[0].shape == (28, 286)
    assert test_data[0].shape == (28, 286)
    assert train_data[1].shape == (28,)
    assert test_data[1].shape == (28,)


def test_load_data_sony_aibo_robot_surface1():
    train_data, test_data = DataLoader('SonyAIBORobotSurface1').load_data()
    assert train_data[0].shape == (20, 70)
    assert test_data[0].shape == (601, 70)
    assert train_data[1].shape == (20,)
    assert test_data[1].shape == (601,)


def test_load_data_fake_Sony_aibo_robot_surface122():
    train_data, test_data = DataLoader('SonyAIBORobotSurface122').load_data()
    assert train_data is None
    assert test_data is None


def test_load_data_lightning7():
    train_data, test_data = DataLoader('Lightning7').load_data()
    assert train_data[0].shape == (70, 319)
    assert test_data[0].shape == (73, 319)
    assert train_data[1].shape == (70,)
    assert test_data[1].shape == (73,)


def test_load_data_umd():
    train_data, test_data = DataLoader('UMD').load_data()
    assert train_data[0].shape == (36, 150)
    assert test_data[0].shape == (144, 150)
    assert train_data[1].shape == (36,)
    assert test_data[1].shape == (144,)
