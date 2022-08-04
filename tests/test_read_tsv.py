from cases.run.utils import read_tsv


def test_read_tsv():
    (X_train, X_test), (y_train, y_test) = read_tsv('ElectricDevices')
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    (X_train, X_test), (y_train, y_test) = read_tsv('EthanolLevel')
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
