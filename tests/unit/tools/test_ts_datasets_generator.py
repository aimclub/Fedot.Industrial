import pytest

from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


@pytest.mark.parametrize('binary, multivariate', [(True, False), (False, True)])
def test_generate_data(binary, multivariate):
    n_samples = 80
    ts_len = 50
    n_classes = 2 if binary else 3
    generator = TimeSeriesDatasetsGenerator(num_samples=80,
                                            max_ts_len=50,
                                            binary=binary,
                                            multivariate=multivariate,
                                            test_size=0.5)
    (X_train, y_train), (X_test, y_test) = generator.generate_data()

    assert X_train.shape[0] + X_test.shape[0] == n_samples * n_classes
    if multivariate:
        assert X_train.iloc[0:1,0:1].values[0][0].shape[0] == ts_len
    else:
        assert X_train.shape[1] == ts_len
